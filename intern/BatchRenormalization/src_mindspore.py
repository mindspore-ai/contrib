import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore import context

class BatchNormalization2D(nn.Cell):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNormalization2D, self).__init__()

        self.eps = Tensor(eps, dtype=mindspore.float32)
        self.momentum = Tensor(momentum, dtype=mindspore.float32)
        self.gamma = Parameter(ops.ones((1, num_features, 1, 1), mindspore.float32))
        self.beta = Parameter(ops.zeros((1, num_features, 1, 1), mindspore.float32))
        self.running_avg_mean = Parameter(ops.zeros((1, num_features, 1, 1), mindspore.float32), requires_grad=False)
        self.running_avg_std = Parameter(ops.ones((1, num_features, 1, 1), mindspore.float32), requires_grad=False)

    def construct(self, x):
        batch_ch_mean = ops.mean(x, axis=(0, 2, 3), keep_dims=True)
        batch_ch_var = ops.mean((x - batch_ch_mean) ** 2, axis=(0, 2, 3), keep_dims=True)
        batch_ch_std = ops.sqrt(batch_ch_var + self.eps)

        if self.training:
            x = (x - batch_ch_mean) / batch_ch_std
            x = x * self.gamma + self.beta
            self.running_avg_mean.set_data((1 - self.momentum) * self.running_avg_mean + self.momentum * batch_ch_mean)
            self.running_avg_std.set_data((1 - self.momentum) * self.running_avg_std + self.momentum * batch_ch_std)
        else:
            running_std = ops.sqrt(self.running_avg_std * self.running_avg_std + self.eps)
            x = (x - self.running_avg_mean) / running_std
            x = self.gamma * x + self.beta

        return x

class BatchRenormalization2D(nn.Cell):
    def __init__(self, num_features, eps=1e-05, momentum=0.01, r_d_max_inc_step=0.0001):
        super(BatchRenormalization2D, self).__init__()

        self.eps = Tensor(eps, dtype=mindspore.float32)
        self.momentum = Tensor(momentum, dtype=mindspore.float32)
        self.gamma = Parameter(ops.ones((1, num_features, 1, 1), mindspore.float32))
        self.beta = Parameter(ops.zeros((1, num_features, 1, 1), mindspore.float32))
        self.running_mean = Parameter(ops.zeros((1, num_features, 1, 1), mindspore.float32), requires_grad=False)
        self.running_std = Parameter(ops.ones((1, num_features, 1, 1), mindspore.float32), requires_grad=False)
        self.max_r_max = Tensor(3.0, dtype=mindspore.float32)
        self.max_d_max = Tensor(5.0, dtype=mindspore.float32)
        self.r_max_inc_step = Tensor(r_d_max_inc_step, dtype=mindspore.float32)
        self.d_max_inc_step = Tensor(r_d_max_inc_step, dtype=mindspore.float32)
        self.r_max = Parameter(Tensor(1.0, dtype=mindspore.float32), requires_grad=False)
        self.d_max = Parameter(Tensor(0.0, dtype=mindspore.float32), requires_grad=False)

    def construct(self, x):
        batch_mean = ops.mean(x, axis=(0, 2, 3), keep_dims=True)
        batch_var = ops.mean((x - batch_mean) ** 2, axis=(0, 2, 3), keep_dims=True)
        batch_std = ops.sqrt(batch_var + self.eps)

        if self.training:
            running_std = ops.sqrt(self.running_std * self.running_std + self.eps)
            r = batch_std / running_std
            d = (batch_mean - self.running_mean) / running_std

            r = ops.clip_by_value(r, 1.0 / self.r_max, self.r_max)
            d = ops.clip_by_value(d, -self.d_max, self.d_max)

            x_hat = (x - batch_mean) / batch_std * r + d
            y = self.gamma * x_hat + self.beta

            batch_size = Tensor(x.shape[0], dtype=mindspore.float32)
            r_max_new = self.r_max + self.r_max_inc_step * batch_size
            d_max_new = self.d_max + self.d_max_inc_step * batch_size
            self.r_max.set_data(ops.minimum(r_max_new, self.max_r_max))
            self.d_max.set_data(ops.minimum(d_max_new, self.max_d_max))

            self.running_mean.set_data((1 - self.momentum) * self.running_mean + self.momentum * batch_mean)
            self.running_std.set_data((1 - self.momentum) * self.running_std + self.momentum * batch_std)
        else:
            running_std = ops.sqrt(self.running_std * self.running_std + self.eps)
            y = self.gamma * (x - self.running_mean) / running_std + self.beta

        return y

# 测试代码
import numpy as np
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
batch_size, channels, height, width = 4, 3, 32, 32
data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
input_tensor = Tensor(data, dtype=mindspore.float32)

bn = BatchNormalization2D(num_features=channels, eps=1e-5, momentum=0.1)
bn.set_train(True)
output_bn_train = bn(input_tensor)
print("BatchNormalization2D (Train Mode) Output Shape:", output_bn_train.shape)
print("BatchNormalization2D (Train Mode) Output Mean:", ops.mean(output_bn_train).asnumpy())
bn.set_train(False)
output_bn_eval = bn(input_tensor)
print("BatchNormalization2D (Eval Mode) Output Shape:", output_bn_eval.shape)
print("BatchNormalization2D (Eval Mode) Output Mean:", ops.mean(output_bn_eval).asnumpy())

brn = BatchRenormalization2D(num_features=channels, eps=1e-5, momentum=0.01, r_d_max_inc_step=0.0001)
brn.set_train(True)
output_brn_train = brn(input_tensor)
print("BatchRenormalization2D (Train Mode) Output Shape:", output_brn_train.shape)
print("BatchRenormalization2D (Train Mode) Output Mean:", ops.mean(output_brn_train).asnumpy())
brn.set_train(False)
output_brn_eval = brn(input_tensor)
print("BatchRenormalization2D (Eval Mode) Output Shape:", output_brn_eval.shape)
print("BatchRenormalization2D (Eval Mode) Output Mean:", ops.mean(output_brn_eval).asnumpy())