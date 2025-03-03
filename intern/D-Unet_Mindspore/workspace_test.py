from DUnet import DUnet
import mindspore
from mindspore import Tensor, context
import mindspore.numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

batch_size = 8

x = Tensor(mindspore.ops.randn(batch_size, 4, 192, 192), mindspore.float32)
model = DUnet(x.shape[1])

output = model(x)

print(output.shape)  # (batch_size, 1, 192, 192)
