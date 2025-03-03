import mindspore
from mindspore import Tensor
import mindspore.numpy as np
from nystrom_attention import Nystromformer

model = Nystromformer(
    dim = 512,
    dim_head = 64,
    heads = 8,
    depth = 6,
    num_landmarks = 256,
    pinv_iterations = 6
)

x = Tensor(mindspore.ops.randn(1, 16384, 512), mindspore.float32)
mask = Tensor(np.ones((1, 16384)), mindspore.bool_)

model(x, mask = mask) # (1, 16384, 512)
print(model.shape)
