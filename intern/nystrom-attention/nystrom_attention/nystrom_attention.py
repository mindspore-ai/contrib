from math import ceil
import mindspore
import numpy as np
from mindspore import nn, ops
from mindspore.ops import functional as F


# help to implement the einops
def swap_last_two(tensor):
    num_dims = tensor.ndim
    new_order = list(range(num_dims))
    new_order[-1], new_order[-2] = new_order[-2], new_order[-1]
    return ops.transpose(tensor, tuple(new_order))


# helper functions
def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    abs_x = ops.Abs()(x)
    col = abs_x.sum(axis=-1)
    row = abs_x.sum(axis=-2)
    z = swap_last_two(x) / (ops.max(col)[0] * ops.max(row)[0])
    I = ops.Eye()(x.shape[-1], x.shape[-1], mindspore.float32)
    mindspore.ops.unsqueeze(I, 0)

    for _ in range(iters):
        xz = ops.matmul(x, z)
        z = 0.25 * ops.matmul(z, (13 * I - ops.matmul(xz, (15 * I - ops.matmul(xz, (7 * I - xz))))))

    return z


# main attention class
class NystromAttention(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.
    ):
        super(NystromAttention, self).__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Dense(dim, inner_dim * 3, has_bias=False)

        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads, heads, (kernel_size, 1), pad_mode='pad',
                padding=(padding, padding, 0, 0), group=heads, has_bias=False
            )

    def construct(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, ((0, 0), (padding, 0), (0, 0)), constant_values=0)

            if exists(mask):
                mask = F.pad(mask, ((padding, 0),), constant_values=False)

        # derive query, keys, values
        qkv = self.to_qkv(x)
        q, k, v = mindspore.ops.chunk(qkv, 3, axis=-1)
        d = q.shape[2] // h
        q = q.reshape(b, n, h, d).transpose(0, 2, 1, 3)  # b n (h d) -> b h n d
        k = k.reshape(b, n, h, d).transpose(0, 2, 1, 3)  # b n (h d) -> b h n d
        v = v.reshape(b, n, h, d).transpose(0, 2, 1, 3)  # b n (h d) -> b h n d

        if exists(mask):
            mask = ops.expand_dims(mask, 1)
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask
        l = ceil(n / m)
        q_landmarks = q.view(b, h, n // l, l, d).sum(axis=-2)
        k_landmarks = k.view(b, h, n // l, l, d).sum(axis=-2)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = mask.view(b, n // l, l).sum(axis=-1)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = np.einsum(einops_eq, q.asnumpy(), k_landmarks.asnumpy())
        sim2 = np.einsum(einops_eq, q_landmarks.asnumpy(), k_landmarks.asnumpy())
        sim3 = np.einsum(einops_eq, q_landmarks.asnumpy(), k.asnumpy())
        sim1 = mindspore.Tensor(sim1, mindspore.float32)
        sim2 = mindspore.Tensor(sim2, mindspore.float32)
        sim3 = mindspore.Tensor(sim3, mindspore.float32)

        # masking
        if exists(mask):
            mask_value = -np.finfo(np.float32).max
            mask_value = mindspore.Tensor(mask_value, mindspore.float32).item()
            sim1 = ops.masked_fill(sim1, ~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2 = ops.masked_fill(sim2, ~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3 = ops.masked_fill(sim3, ~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values
        attn1, attn2, attn3 = map(lambda t: ops.softmax(t, axis=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out1 = ops.matmul(attn1, attn2_inv)
        out2 = ops.matmul(attn3, v)
        out = ops.matmul(out1, out2)

        # add depth-wise conv residual of values
        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads
        out = out.transpose(0, 2, 1, 3).view(b, n, h * d)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = ops.matmul(ops.matmul(attn1, attn2_inv), attn3)
            return out, attn

        return out


# transformer
class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn

    def construct(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Cell):
    def __init__(self, dim, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Dense(dim * mult, dim)
        )

    def construct(self, x):
        return self.net(x)


class Nystromformer(nn.Cell):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super(Nystromformer, self).__init__()

        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                PreNorm(dim, NystromAttention(
                    dim=dim, dim_head=dim_head, heads=heads, num_landmarks=num_landmarks,
                    pinv_iterations=pinv_iterations, residual=attn_values_residual,
                    residual_conv_kernel=attn_values_residual_conv_kernel, dropout=attn_dropout
                )),
                PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout))
            ]))

    def construct(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x
