import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import XavierUniform

class NAC(nn.Cell):
    def __init__(self, in_dim, out_dim):
        super(NAC, self).__init__()
        self.W_hat = ms.Parameter(
            Tensor(np.random.uniform(-1, 1, size=(out_dim, in_dim)), dtype=ms.float32))
        self.M_hat = ms.Parameter(
            Tensor(np.random.uniform(-1, 1, size=(out_dim, in_dim)), dtype=ms.float32))
        self.W = ops.Tanh()(self.W_hat) * ops.Sigmoid()(self.M_hat)

    def construct(self, x):
        return ops.matmul(x, self.W.T)


class StackedNAC(nn.Cell):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim):
        super(StackedNAC, self).__init__()
        layers = []
        for i in range(n_layers):
            layer_in_dim = in_dim if i == 0 else hidden_dim
            layer_out_dim = out_dim if i == n_layers - 1 else hidden_dim
            layers.append(NAC(layer_in_dim, layer_out_dim))
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        return self.model(x)


class NALU(nn.Cell):
    def __init__(self, in_dim, out_dim):
        super(NALU, self).__init__()
        self.G = nn.Dense(in_dim, out_dim, weight_init=XavierUniform(), has_bias=True)
        self.nac = NAC(in_dim, out_dim)
        self.eps = 1e-7

    def construct(self, x):
        a = self.nac(x)
        g = ops.Sigmoid()(self.G(x))
        log_input = ops.Log()(ops.Abs()(x) + self.eps)
        m = ops.Exp()(self.nac(log_input))
        y = g * a + (1 - g) * m
        return y


class StackedNALU(nn.Cell):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim):
        super(StackedNALU, self).__init__()
        layers = []
        for i in range(n_layers):
            layer_in_dim = in_dim if i == 0 else hidden_dim
            layer_out_dim = out_dim if i == n_layers - 1 else hidden_dim
            layers.append(NALU(layer_in_dim, layer_out_dim))
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        return self.model(x)