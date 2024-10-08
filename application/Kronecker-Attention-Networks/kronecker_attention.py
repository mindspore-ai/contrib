from mindspore import nn, ops, Tensor


class KroneckerSelfAttention(nn.Cell):
    def __init__(self, dim, heads, dim_heads=32):
        super().__init__()
        hidden_dim = heads * dim_heads

        self.heads = heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, has_bias=False, pad_mode='valid')
        self.to_out = nn.Conv1d(hidden_dim, dim, 1, pad_mode='valid')

    def construct(self, x: Tensor):
        h_last = x.shape[-2]
        x = ops.cat((x.mean(axis=-1), x.mean(axis=-2)), axis=-1)
        qkv = self.to_qkv(x)
        b, qkv_h_d, n = qkv.shape
        h = self.heads
        qkv_value = 3
        d = qkv_h_d // (qkv_value * h)
        qkv_reshaped = ops.reshape(qkv, (b, qkv_value, h, d, n))
        qkv_transposed = ops.transpose(qkv_reshaped, (1, 0, 2, 3, 4))
        q, k, v = qkv_transposed[0], qkv_transposed[1], qkv_transposed[2]
        dots = ops.einsum('bhdi,bhdj->bhij', q, k)
        attn = ops.softmax(dots, axis=-1)
        out = ops.einsum('bhij,bhdj->bhdi', attn, v)
        b, h, d, n = out.shape
        out = ops.reshape(out, (b, h * d, n))
        out = self.to_out(out)
        out1 = out[..., :h_last].unsqueeze(3)
        out2 = out[..., h_last:].unsqueeze(2)
        out = out1 + out2
        return out


if __name__ == '__main__':
    attn = KroneckerSelfAttention(
        dim=32,
        heads=8,
        dim_heads=64
    )
    x = ops.randn(1, 32, 256, 512)
    print(f"output shape {attn(x).shape}")  # (1, 32, 256, 512)
