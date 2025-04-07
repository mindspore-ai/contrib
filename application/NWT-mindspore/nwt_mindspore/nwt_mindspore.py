import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter


class EinMix(nn.Cell):
    """MindSpore实现的EinMix，模拟PyTorch的EinMix功能"""

    def __init__(self, pattern, weight_shape, h, d, c):
        super().__init__()
        self.h = h
        self.d = d
        self.c = c
        # 参数初始化
        weight_shape = (h, d, c)
        self.weight = Parameter(Tensor(np.random.randn(*weight_shape).astype(np.float32)))

        # 定义常用操作
        self.matmul = ops.BatchMatMul()
        self.reshape = ops.Reshape()

    def construct(self, x):
        # h n d -> h n c
        # 实现einsum: h n d, h d c -> h n c
        x_reshaped = self.reshape(x, (self.h, -1, self.d))
        result = self.matmul(x_reshaped, self.weight)
        return result


class Memcodes(nn.Cell):
    def __init__(
            self,
            *,
            dim,
            num_codes,
            heads=8,
            temperature=1.,
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.temperature = temperature
        self.num_codes = num_codes

        num_codebooks = heads
        codebook_dim = dim // heads

        # 创建参数
        self.codes = Parameter(Tensor(np.random.randn(num_codebooks, num_codes, codebook_dim).astype(np.float32)))
        self.to_k = EinMix('h n d -> h n c', weight_shape='h d c', h=heads, d=codebook_dim, c=codebook_dim)
        self.to_v = EinMix('h n d -> h n c', weight_shape='h d c', h=heads, d=codebook_dim, c=codebook_dim)

        # 定义操作
        self.reshape = ops.Reshape()
        self.expand_dims = ops.ExpandDims()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        self.gather = ops.GatherD()
        self.argmax = ops.Argmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()
        self.one_hot = ops.OneHot()
        self.cast = ops.Cast()
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)
        self.stack = ops.Stack()
        self.squeeze = ops.Squeeze()

    def construct(self, x, *, merge_output_heads=True):
        assert x.shape[-1] == self.dim, f"Input dimension {x.shape[-1]} doesn't match expected dimension {self.dim}"

        # 拆分头: b n (h d) -> b h n d
        b, n, _ = x.shape
        h = self.heads
        d = self.dim // h
        q = self.reshape(x, (b, n, h, d))
        q = self.transpose(q, (0, 2, 1, 3))  # b h n d
        q = q * self.scale

        # 获取编码本的键值
        k = self.to_k(self.codes)  # h codes d
        v = self.to_v(self.codes)  # h codes d

        # 计算注意力分数: b h n codes
        # 使用batch_matmul替代复杂的einsum实现
        q_reshaped = self.reshape(q, (b * h, n, d))  # (b*h, n, d)
        k_expanded = self.tile(self.expand_dims(k, 0), (b, 1, 1, 1))  # (b, h, codes, d)
        k_reshaped = self.reshape(k_expanded, (b * h, self.num_codes, d))  # (b*h, codes, d)
        k_transposed = self.transpose(k_reshaped, (0, 2, 1))  # (b*h, d, codes)

        logits = self.batch_matmul(q_reshaped, k_transposed)  # (b*h, n, codes)
        logits = self.reshape(logits, (b, h, n, self.num_codes))  # (b, h, n, codes)

        # 根据训练状态使用不同的方法
        if self.training:
            # 简化Gumbel-softmax实现，直接使用softmax
            attn = self.softmax(logits / self.temperature)
            codebook_indices = self.argmax(attn)
        else:
            codebook_indices = self.argmax(logits)
            attn = self.cast(
                self.one_hot(codebook_indices, self.num_codes,
                             ops.scalar_to_tensor(1.0),
                             ops.scalar_to_tensor(0.0)),
                ms.float32
            )

        # 计算输出: b h n d
        # 简化einsum的实现，直接使用批量矩阵乘法
        v_expanded = self.tile(self.expand_dims(v, 0), (b, 1, 1, 1))  # (b, h, codes, d)

        # 重塑张量以进行批量矩阵乘法
        attn_reshaped = self.reshape(attn, (b * h, n, self.num_codes))  # (b*h, n, codes)
        v_reshaped = self.reshape(v_expanded, (b * h, self.num_codes, d))  # (b*h, codes, d)

        out = self.batch_matmul(attn_reshaped, v_reshaped)  # (b*h, n, d)
        out = self.reshape(out, (b, h, n, d))  # (b, h, n, d)

        if not merge_output_heads:
            return out, codebook_indices

        # 合并头: b h n d -> b n (h d)
        out = self.transpose(out, (0, 2, 1, 3))  # (b, n, h, d)
        out = self.reshape(out, (b, n, h * d))  # (b, n, h*d)

        return out, codebook_indices

    def get_codes_from_indices(self, codebook_indices, *, merge_output_heads=True):
        """从编码索引重建输出"""
        # 确定形状
        batch, seq_len = codebook_indices.shape[0], codebook_indices.shape[2]
        h = self.heads
        d = self.dim // h

        # 重塑编码索引，确保形状正确
        if codebook_indices.shape[1] == h and codebook_indices.shape[2] == seq_len:
            # 形状已经是(batch, heads, seq)，无需调整
            indices = codebook_indices
        else:
            # 需要转置为(batch, heads, seq)
            indices = self.transpose(codebook_indices, (0, 2, 1))

        # 获取编码本值
        values = self.to_v(self.codes)  # (h, num_codes, d)

        # 为每个批次扩展编码本
        expanded_values = self.tile(
            self.expand_dims(values, 0),
            (batch, 1, 1, 1)
        )  # (batch, h, codes, d)

        # 为每个位置创建one-hot向量
        one_hot_indices = self.cast(
            self.one_hot(indices, self.num_codes,
                         ops.scalar_to_tensor(1.0),
                         ops.scalar_to_tensor(0.0)),
            ms.float32
        )  # (batch, h, seq, codes)

        # 重塑张量以进行批量矩阵乘法
        one_hot_reshaped = self.reshape(one_hot_indices, (batch * h, seq_len, self.num_codes))  # (batch*h, seq, codes)
        values_reshaped = self.reshape(expanded_values, (batch * h, self.num_codes, d))  # (batch*h, codes, d)

        # 批量矩阵乘法获取编码值
        out = self.batch_matmul(one_hot_reshaped, values_reshaped)  # (batch*h, seq, d)
        out = self.reshape(out, (batch, h, seq_len, d))  # (batch, h, seq, d)

        if not merge_output_heads:
            return out

        # 合并头: (batch, h, seq, d) -> (batch, seq, h*d)
        out = self.transpose(out, (0, 2, 1, 3))  # (batch, seq, h, d)
        out = self.reshape(out, (batch, seq_len, h * d))  # (batch, seq, h*d)

        return out