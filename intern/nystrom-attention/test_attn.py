import mindspore
from mindspore import Tensor
import mindspore.numpy as np
from nystrom_attention import NystromAttention

attn = NystromAttention(
    dim = 512,
    dim_head = 64,
    heads = 8,
    num_landmarks = 256,    # number of landmarks
    pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
    residual = True         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
)
x = Tensor(mindspore.ops.randn(1, 16384, 512), mindspore.float32)
mask = Tensor(np.ones((1, 16384)), mindspore.bool_)

attn(x, mask = mask) # (1, 16384, 512)
