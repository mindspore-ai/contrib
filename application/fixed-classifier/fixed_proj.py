import math
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ops
from mindspore.common.initializer import initializer
from scipy.linalg import hadamard


class HadamardProj(nn.Cell):

    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_scale=None):
        super(HadamardProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = Tensor(hadamard(sz), ms.float32)
        self.proj = Parameter(mat, requires_grad=not fixed_weights)

        init_scale = 1. / math.sqrt(self.output_size)

        self.scale = Parameter(Tensor([init_scale], dtype=ms.float32), requires_grad=(fixed_scale is None))

        if bias:
            self.bias = Parameter(ops.uniform((output_size, ), Tensor(-init_scale, dtype=ms.float32), Tensor(init_scale, dtype=ms.float32)))
        else:
            self.bias = None

        self.eps = 1e-8

    def construct(self, x):
        x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)

        out = -self.scale * \
            ops.dense(x, w[:self.output_size, :self.input_size])
            
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class Proj(nn.Cell):

    def __init__(self, input_size, output_size, bias=True, init_scale=10):
        super(Proj, self).__init__()
        if init_scale is not None:
            self.weight = Parameter(ops.fill(ms.float32, (1, ), init_scale))
        if bias:
            self.bias = Parameter(ops.zeros(output_size))
        ms.set_seed(123)
        self.proj = Parameter(initializer('orthogonal', (output_size, input_size)), requires_grad=False)

    def construct(self, x):
        w = self.proj.type_as(x)
        x = x / x.norm(2, -1, keepdim=True)
        out = ops.dense(x, w)
        if hasattr(self, 'weight'):
            out = out * self.weight
        if hasattr(self, 'bias'):
            out = out + self.bias.view(1, -1)
        return out
    
    
if __name__ == '__main__':
    import numpy as np
    
    x = Tensor(np.random.randn(2, 3), ms.float32)
    
    net = HadamardProj(3, 5)
    output = net(x)
    print(output)
    print(output.shape)
    
    net = Proj(3, 5)
    output = net(x)
    print(output)
    print(output.shape)
