import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal
import numpy as np


class RelativeAttentionBias(nn.Cell):
    def __init__(self, num_heads, relative_attention_num_buckets, relative_attention_max_distance=128):
        super(RelativeAttentionBias, self).__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.cast = ops.Cast()
        self.abs = ops.Abs()
        self.minimum = ops.Minimum()
        self.log = ops.Log()
        self.floor = ops.Floor()
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.less = ops.Less()
        self.greater = ops.Greater()
        self.dtype = mindspore.int32

    def construct(self, query_length, key_length):
        context_position = Tensor(np.arange(query_length), mindspore.int32).view(-1, 1)
        memory_position = Tensor(np.arange(key_length), mindspore.int32).view(1, -1)
        relative_position = memory_position - context_position  # 形状 (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # 形状 (query_length, key_length, num_heads)
        values = self.transpose(values, (2, 0, 1))
        values = self.expand_dims(values, 0)  # 形状 (1, num_heads, query_length, key_length)
        return values

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = self.zeros_like(relative_position)
        num_buckets_tensor = Tensor(num_buckets, mindspore.int32)
        if bidirectional:
            num_buckets = num_buckets // 2
            num_buckets_tensor = num_buckets_tensor // 2
            greater_than_zero = self.cast(self.greater(relative_position, 0), mindspore.int32)
            relative_buckets += greater_than_zero * num_buckets
            relative_position = self.abs(relative_position)
        else:
            zeros_tensor = self.zeros_like(relative_position)
            relative_position = -self.minimum(relative_position, zeros_tensor)
        max_exact = num_buckets // 2
        is_small = self.less(relative_position, Tensor(max_exact, mindspore.int32))

        # 计算较大的相对位置的桶
        relative_position_if_large = self.cast(relative_position, mindspore.float32)
        relative_position_if_large = relative_position_if_large / max_exact
        relative_position_if_large = self.log(relative_position_if_large)
        relative_position_if_large = relative_position_if_large / math.log(max_distance / max_exact)
        relative_position_if_large = relative_position_if_large * (num_buckets - max_exact)
        relative_position_if_large = relative_position_if_large + max_exact
        relative_position_if_large = self.cast(self.floor(relative_position_if_large), mindspore.int32)
        num_buckets_minus_1 = Tensor(num_buckets - 1, mindspore.int32)
        relative_position_if_large = self.minimum(relative_position_if_large, num_buckets_minus_1)

        relative_position = self.select(is_small, relative_position, relative_position_if_large)
        relative_buckets += relative_position
        return relative_buckets

class PointwiseAggregatedAttention(nn.Cell):
    def __init__(self, d_model, num_heads):
        super(PointwiseAggregatedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.rab_p = RelativeAttentionBias(num_heads, relative_attention_num_buckets=32,
                                           relative_attention_max_distance=128)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.matmul = ops.BatchMatMul()
        self.silu = nn.SiLU()
        self.flatten = ops.Flatten()
        self.expand_dims = ops.ExpandDims()
        self.shape = ops.Shape()

    def split_heads(self, x):
        batch_size, seq_length, d_model = self.shape(x)
        x = self.reshape(x, (batch_size, seq_length, self.num_heads, self.head_dim))
        x = self.transpose(x, (0, 2, 1, 3))
        return x

    def construct(self, v, k, q, mask=None):
        batch_size = q.shape[0]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        k_t = self.transpose(k, (0, 1, 3, 2))
        attention_scores = self.matmul(q, k_t)
        # attention_scores = attention_scores / math.sqrt(self.head_dim)
        rab = self.rab_p(q.shape[2], k.shape[2])  # 形状 (1, num_heads, query_length, key_length)

        att_w_bias = attention_scores + rab

        av = self.matmul(self.silu(att_w_bias), v)
        av = self.transpose(av, (0, 2, 1, 3))
        av = self.reshape(av, (batch_size, -1, self.num_heads * self.head_dim))
        return av

class HSTUBlock(nn.Cell):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(HSTUBlock, self).__init__()
        self.f1 = nn.Dense(d_model, d_model * 4)
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, num_heads)
        self.f2 = nn.Dense(d_model, d_model)
        self.norm = nn.LayerNorm((d_model,))
        self.split = ops.Split(axis=-1, output_num=4)
        self.silu = nn.SiLU()

    def construct(self, x):
        x_proj = self.silu(self.f1(x))
        u, v, q, k = self.split(x_proj)
        av = self.pointwise_attn(v, k, q)
        y = self.f2(self.norm(av * u))
        return y

class GenRec(nn.Cell):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super(GenRec, self).__init__()
        self.layers = nn.CellList([HSTUBlock(d_model, num_heads, dropout) for _ in range(num_layers)])

    def construct(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x

if __name__ == "__main__":
    model = GenRec(d_model=52, num_heads=2, num_layers=3)
    input_shape = (32, 10, 52)
    x = Tensor(np.random.rand(*input_shape), mindspore.float32)
    output = model(x)
    assert output.shape == x.shape
