import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.context
import numpy as np

class FCT(nn.Cell):
    def __init__(self, dim=2048, decode_dim=1024, hw=400):
        super(FCT, self).__init__()
        self.dim_o = dim
        a = dim
        dim, decode_dim = hw, hw
        hw = a
        self.decode_dim = decode_dim
        self.weight_q = nn.Dense(dim, decode_dim, has_bias=False)
        self.weight_k = nn.Dense(dim, decode_dim, has_bias=False)
        self.weight_alpha = Parameter(Tensor(np.random.randn(hw // 2 + 1, hw // 2 + 1) * 0.02, mindspore.float32))
        self.proj = nn.Dense(hw, hw)
        self.ac_bn_2 = nn.SequentialCell([nn.ReLU(), nn.BatchNorm2d(self.dim_o)])
        #self.writer
        
    def construct(self, x):
        raw = x
        B, C, H, W = x.shape
        N = H * W
        
        x = x.reshape(B, N, C)
        q = self.weight_q(x)
        q = ops.swapaxes(q,-2,-1) # [B, N, C]
        k = self.weight_k(x)
        k = ops.swapaxes(k,-2,-1) # [B, N, C]
        fft = ops.FFTWithSize(signal_ndim=3, inverse=False, real=True, norm='ortho')
        q = fft(q)  
        k = fft(k)  
        
        q_r, q_i = q.real(), q.imag()
        q_r = ops.swapaxes(q_r,-2,-1)
        q_i = ops.swapaxes(q_i,-2,-1)
        attn_r = q_r@ k.real()
        attn_i = q_i@ k.imag()
        attn_r = self.weight_alpha * attn_r
        attn_i = self.weight_alpha * attn_i
        x_r = logmax(attn_r) @ q_i  # [B, N, C]
        x_i = logmax(attn_i) @ q_r  # [B, N, C]
        
        stacked = ops.stack([x_r, x_i],-1)
        x = ops.Complex()(stacked[...,0], stacked[...,1])
        x = ops.swapaxes(x,-2,-1)
        
        irfft = ops.FFTWithSize(signal_ndim=3, inverse=True, real=True, norm='ortho')
        x = ops.irfft(x)  
        x = self.proj(x)
        x = x.reshape(B, C, H, W)
        x = self.ac_bn_2(x)
        
        return raw + x
    
def logmax(X, axis=-1):
    X_log = ops.log(X+1e-9)
    X = ops.sum(X_log, axis, keepdim=True)
    return X_log / X
        
if __name__ == "__main__":
    B = 2 
    C = 400 
    H = W = 20  
    x = Tensor(np.random.randn(B, C, H, W).astype(np.float32))
    model = FCT(dim=400, decode_dim=1024, hw=400)
    output = model(x)
    print("output.shape:", output.shape)