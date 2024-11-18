import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal
from math import pi

class FourierNet(nn.Cell):
    def __init__(self, num_layers = 4, num_units = 256, fourier = True):
        self.num_layers = num_layers
        self.num_units = num_units
        super(FourierNet, self).__init__()

        relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #fourier option
        if fourier:
            self.B = nn.Dense(2, num_units)
            self.B.weight = mindspore.Parameter(initializer(Normal(sigma = 10.0), self.B.weight.shape))
            self.B.weight.requires_grad = False
        layers = []
        for layer in range(self.num_layers):
            if layer == 0:
                layers.append(nn.Dense(2 * num_units, num_units))
            elif layer == self.num_layers - 1:
                layers.append(nn.Dense(num_units, 3))
            else:
                layers.append(nn.Dense(num_units, num_units))

            if layer != self.num_layers - 1:
                layers.append(relu)
            else:
                layers.append(self.sigmoid)

        self.layers = nn.SequentialCell(*layers)


    def fourier_map(self, x):
        sinside = ops.sin(2 * pi * self.B(x))
        cosside = ops.cos(2 * pi * self.B(x))
        return ops.concat([sinside, cosside], -1)


    def construct(self, x):
        b, c, h, w = x.shape
        x = x.permute((0, 2, 3, 1)).view((b, -1, c))
        x = self.fourier_map(x)
        r = self.layers(x)
        return r.view((b, 3, h, w))

