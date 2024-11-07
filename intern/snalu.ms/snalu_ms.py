import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import XavierUniform
from mindspore import context

class NAC(nn.Cell):
    def __init__(self, in_dim, out_dim):
        super(NAC, self).__init__()
        self.W_hat = Parameter(Tensor(np.random.uniform(-1, 1, size=(out_dim, in_dim)), dtype=ms.float32), name='W_hat')
        self.M_hat = Parameter(Tensor(np.random.uniform(-1, 1, size=(out_dim, in_dim)), dtype=ms.float32), name='M_hat')
        self.tanh = ops.Tanh()
        self.sigmoid = ops.Sigmoid()
        self.matmul = ops.MatMul()

    def construct(self, x):
        W = self.tanh(self.W_hat) * self.sigmoid(self.M_hat)
        return self.matmul(x, W.T)
    
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
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.abs = ops.Abs()
        self.exp = ops.Exp()
        self.mul = ops.Mul()

    def construct(self, x):
        a = self.nac(x)
        g = self.sigmoid(self.G(x))
        log_input = self.log(self.abs(x) + self.eps)
        m = self.exp(self.nac(log_input))
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
class SNAC(nn.Cell):
    def __init__(self, in_dim, out_dim):
        super(SNAC, self).__init__()
        self.W_hat = Parameter(Tensor(np.random.uniform(-1, 1, size=(out_dim, in_dim)), dtype=ms.float32), name='W_hat')
        self.M_hat = Parameter(Tensor(np.random.uniform(-1, 1, size=(out_dim, in_dim)), dtype=ms.float32), name='M_hat')
        self.tanh = ops.Tanh()
        self.sigmoid = ops.Sigmoid()
        self.matmul = ops.MatMul()

    def construct(self, x):
        W = self.tanh(self.W_hat) * self.sigmoid(self.M_hat)
        return self.matmul(x, W.T)
        
class SNALU(nn.Cell):
    def __init__(self, in_dim, out_dim):
        super(SNALU, self).__init__()
        self.G = nn.Dense(in_dim, out_dim, weight_init=XavierUniform(), has_bias=True)
        self.snac = SNAC(in_dim, out_dim)
        self.eps = 1e-7
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.abs = ops.Abs()
        self.exp = ops.Exp()
        self.mul = ops.Mul()

    def construct(self, x):
        a = self.snac(x)
        g = self.sigmoid(self.G(x))
        log_input = self.log(self.abs(x) + self.eps)
        m = self.exp(self.snac(log_input))
        y = g * a + (1 - g) * m
        return y
        
class StackedSNALU(nn.Cell):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim):
        super(StackedSNALU, self).__init__()
        layers = []
        for i in range(n_layers):
            layer_in_dim = in_dim if i == 0 else hidden_dim
            layer_out_dim = out_dim if i == n_layers - 1 else hidden_dim
            layers.append(SNALU(layer_in_dim, layer_out_dim))
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        return self.model(x)
