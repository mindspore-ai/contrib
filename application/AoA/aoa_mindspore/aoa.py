import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class GLU(nn.Cell):
    def __init__(self, axis=-1):
        super(GLU, self).__init__()
        self.axis = axis
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        a, b = ops.split(x, x.shape[self.axis] // 2, self.axis)
        return a * self.sigmoid(b)


class AttentionOnAttention(nn.Cell):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0., aoa_dropout=0.):
        super(AttentionOnAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Dense(dim, inner_dim, has_bias=False)
        self.to_kv = nn.Dense(dim, inner_dim * 2, has_bias=False)

        self.dropout = nn.Dropout(keep_prob=1 - dropout)

        self.aoa = nn.SequentialCell([
            nn.Dense(2 * inner_dim, 2 * dim),
            GLU(),
            nn.Dropout(keep_prob=1 - aoa_dropout)
        ])

    def construct(self, x, context=None):
        h = self.heads

        q_ = self.to_q(x)

        context = default(context, x)
        kv = ops.split(self.to_kv(context), self.dim_head * self.heads, -1)
        
        # split heads
        def reshape_to_heads(t):
            b, n, _ = t.shape
            return t.reshape(b, n, h, self.dim_head).transpose(0, 2, 1, 3)

        q = reshape_to_heads(q_)
        k = reshape_to_heads(kv[0])
        v = reshape_to_heads(kv[1])

        # q: (b, h, n, d)，   k: (b, h, n, d) -> k_t(b, h, d, n)
        k_t = k.transpose(0, 1, 3, 2)
        dots = ops.matmul(q, k_t) * self.scale

        # attention
        attn = ops.Softmax(axis=-1)(dots)
        attn = self.dropout(attn)

        # weighted average of values
        attn_out = ops.matmul(attn, v)  # shape: (b, h, n, d)

        # concat heads (b, n, h, d) -> (b, n, h*d)
        out = attn_out.transpose(0, 2, 1, 3)
        b, n, _, _ = out.shape
        out = out.reshape(b, n, h * self.dim_head)

        # attention on attention
        out = self.aoa(ops.concat((out, q_), axis=-1))
        return out
