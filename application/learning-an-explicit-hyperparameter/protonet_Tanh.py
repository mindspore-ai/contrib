import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal, Constant

class ConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, retain_activation=True):
        super(ConvBlock, self).__init__()

        conv_weight_init = Normal(sigma=0.02)
        bn_gamma_init = Constant(1)
        bn_beta_init = Constant(0)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False, weight_init=conv_weight_init)
        self.bn = nn.BatchNorm2d(out_channels, gamma_init=bn_gamma_init, beta_init=bn_beta_init)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.activation = self.tanh if retain_activation else self.relu

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.max_pool2d(x)
        return x

class ProtoNetEmbedding(nn.Cell):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True):
        super(ProtoNetEmbedding, self).__init__()

        self.encoder = nn.SequentialCell([
            ConvBlock(x_dim, h_dim),
            ConvBlock(h_dim, h_dim),
            ConvBlock(h_dim, h_dim),
            ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation)
        ])

    def construct(self, x):
        x = self.encoder(x)
        return x.reshape(x.shape[0], -1)

if __name__ == "__main__":
    model = ProtoNetEmbedding()
    input = ms.Tensor(np.random.rand(1, 3, 28, 28), ms.float32)
    output = model(input)
    print("output=",output)
    print("output.shape=",output.shape)