import numpy as np
from mindspore import Tensor
from aoa import AttentionOnAttention as AoA

attn = AoA(
    dim=512,
    heads=8
)

x = Tensor(np.random.randn(1, 1024, 512).astype(np.float32))
context = Tensor(np.random.randn(1, 1024, 512).astype(np.float32))
y = attn(x, context=context) + x  # (1, 1024, 512)
print(y.shape)