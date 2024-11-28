import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor,context
import numpy as np

from pixelshuffle1d import PixelShuffle1D, PixelUnshuffle1D

def conv1d_same(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    # 1D Convolution which does not change input size
    # "same" padding
    return nn.Conv1d(in_channels, out_channels, kernel_size, pad_mode = 'pad',
                            padding=(kernel_size-1)//2, has_bias=bias, dilation=dilation)

#minspore暂不支持linear插值，手动实现如下：
def linear_interpolate_1d(x, scale_factor):
    batch_size, channels, length = x.shape
    target_length = int(length * scale_factor)
    output = np.zeros((batch_size, channels, target_length), dtype=np.float32)

    for i in range(target_length):
        # 映射回输入张量的浮点位置
        src_idx = i / scale_factor
        left_idx = int(np.floor(src_idx))
        right_idx = min(left_idx + 1, length - 1)

        # 计算线性权重
        alpha = src_idx - left_idx
        output[:, :, i] = (1 - alpha) * x[:, :, left_idx] + alpha * x[:, :, right_idx]
    return Tensor(output, mindspore.float32)


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

batch_size = 8
channel_len = 2
sample_len = 44100

x = ops.rand(batch_size, channel_len, sample_len)     # input
scale_factor = 2.0

# Use pixelshuffler as a module
pixel_upsample = PixelShuffle1D(scale_factor)
pixel_downsample = PixelUnshuffle1D(scale_factor)

# Check if PixelUnshuffle1D is the inverse of PixelShuffle1D
x_up = pixel_upsample(x)
x_up_down = pixel_downsample(x_up)

if ops.all(ops.equal(x, x_up_down)):
    print('Inverse module works.')



#完整复现原仓库中linear插值的情况，手动实现，速度较慢
#t = linear_interpolate_1d(x, scale_factor)
#对精度要求不高情况下，使用nearest代替
t = ops.interpolate(x, scale_factor=scale_factor, mode='nearest', recompute_scale_factor=True)

n_conv_ch = 512
kernel_conv = 5

net = nn.SequentialCell(
                    pixel_upsample,
                    conv1d_same(int(channel_len//scale_factor), n_conv_ch, kernel_conv),
                    nn.ReLU(),
                    conv1d_same(n_conv_ch, channel_len, kernel_conv)
                    )

loss_func = nn.MSELoss()
optim = nn.Adam(net.trainable_params(), learning_rate=1e-5)

def get_loss(x,t):
    y = net(x)   
    loss = loss_func(y, t) 
    return loss
grad_fn = mindspore.value_and_grad(get_loss, None, optim.parameters)


for _ in range(500):
    loss,grads = grad_fn(x,t)
    optim(grads)
    print('Loss:',loss)
