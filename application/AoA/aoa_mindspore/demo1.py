import numpy as np
import mindspore
from mindspore import Tensor
from aoa import AttentionOnAttention as AoA

attn = AoA(
    dim = 512,
    heads = 8
)

x = Tensor(np.random.randn(1, 1024, 512), mindspore.float32)
y = attn(x) + x   # (1, 1024, 512)
print(y.shape)
