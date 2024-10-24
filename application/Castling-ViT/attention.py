import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


ms.set_context(device_target='Ascend', device_id=0)


class MatMul(nn.Cell):
    def __init__(self):
        super().__init__()
    def construct(self, a, b):
        out = ops.matmul(a, b)
        return out
    
class LinAngularAttention(nn.Cell):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5
        self.sparse_reg = sparse_reg
        
        self.qkv = nn.Dense(in_channels, in_channels * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(in_channels, in_channels)
        self.proj_drop = nn.Dropout(p=proj_drop)
        
        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()
        
        self.pad = nn.Pad(paddings=((0,0), (0,0), (res_kernel_size // 2, res_kernel_size // 2), (0,0)), mode="CONSTANT")
        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=0,
            pad_mode='valid',
            has_bias=False,
            group=self.num_heads,
        )
        
    def construct(self, x):
        N, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        
        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(0, 1, 3, 2))
            attn = attn.softmax(axis=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn
            
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        v_pad = self.pad(v)
        dconv_v = self.dconv(v_pad)                  
        attn = self.kq_matmul(k.transpose((0, 1, 3, 2)), v)
        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose((0, 1, 3, 2)).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

def main():
    input = ops.randn(1, 196, 256)

    linear_angular_attention = LinAngularAttention(in_channels=256, num_heads=8, qkv_bias=False, sparse_reg=False)
    output = linear_angular_attention(input)
    print(output.shape)

    # linear angular attention with DWConv + SparseAttn
    linear_angular_attention = LinAngularAttention(in_channels=256, num_heads=8, qkv_bias=False, sparse_reg=True)
    output = linear_angular_attention(input)
    print(output.shape)
    

if __name__ == "__main__":
    main()