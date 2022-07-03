"""
decoder注意力模块
"""
from mindspore import nn, Tensor
from mindspore.ops import operations as P

from mindvision.check_param import Validator


class MyAttention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = MyAttention(768, 12)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(MyAttention, self).__init__()
        Validator.check_equal_int(dim % num_heads, 0, 'dim should be divisible by num_heads.')
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(head_dim ** -0.5)
        self.q = nn.Dense(dim, dim)
        self.kv = nn.Dense(dim, dim * 2)
        self.qkv = P.Concat(axis=2)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.mul = P.Mul()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.unstack = P.Unstack(axis=0)
        self.attn_matmul_v = P.BatchMatMul()
        self.q_matmul_k = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x1, x2):
        """Attention construct."""
        b_q, n_q, c_q = x1.shape
        b, n, c = x2.shape
        q = self.q(x1)
        kv = self.kv(x2)
        print(q.shape, kv.shape)
        # qkv = self.qkv((q,kv))
        q = self.reshape(q, (b_q, n_q, self.num_heads, c_q // self.num_heads))
        q = self.transpose(q, (0, 2, 1, 3))
        kv = self.reshape(kv, (b, n, 2, self.num_heads, c // self.num_heads))
        kv = self.transpose(kv, (2, 0, 3, 1, 4))
        k, v = self.unstack(kv)

        # print(q.shape,k.shape,v.shape)
        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b_q, n_q, c_q))
        out = self.out(out)
        out = self.out_drop(out)
        # print(out.shape)
        return out


if __name__ == '__main__':
    attention = MyAttention(dim=768)
    print(attention)
