# coding=utf-8

from mindspore import nn
from mindspore.ops import operations as P

__all__ = ['get_eprnet', 'EPRNet']


class EPRNet(nn.Cell):
    def __init__(self, nclass, activation='relu', drop_out=.1, phase='train'):
        super(EPRNet, self).__init__()
        self.phase = phase
        self.shape = P.Shape()
        self.stem = nn.Conv2dBnAct(3, 16, 3, 2, 'pad', 1,
                                   weight_init='xavier_uniform',
                                   has_bn=True,
                                   activation=activation)
        self.layer1 = nn.SequentialCell([
            _EPRModule(16, 32, atrous_rates=(1, 2, 4), activation=activation),
            _EPRModule(32, 32, atrous_rates=(1, 2, 4), activation=activation, down_sample=True)
        ])
        self.layer2 = nn.SequentialCell([
            _EPRModule(32, 64, atrous_rates=(3, 6, 9), activation=activation),
            _EPRModule(64, 64, atrous_rates=(3, 6, 9), activation=activation),
            _EPRModule(64, 64, atrous_rates=(3, 6, 9), activation=activation),
            _EPRModule(64, 64, atrous_rates=(3, 6, 9), activation=activation, down_sample=True)
        ])
        self.layer3 = nn.SequentialCell([
            _EPRModule(64, 128, atrous_rates=(7, 13, 19), activation=activation),
            _EPRModule(128, 128, atrous_rates=(13, 25, 37), activation=activation)
        ])
        self.stern_conv1 = nn.Conv2dBnAct(128, 128, 3, 1, 'pad', 1,
                                          weight_init='xavier_uniform',
                                          has_bn=True,
                                          activation=activation)
        self.stern_drop = nn.Dropout(drop_out) if drop_out else None
        self.stern_conv2 = nn.Conv2d(128, nclass, 1, weight_init='xavier_uniform')

    def construct(self, x):
        size = self.shape(x)

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.stern_conv1(out)
        if self.stern_drop and self.phase == 'train':
            out = self.stern_drop(out)
        out = self.stern_conv2(out)

        # out = P.ResizeBilinear((size[2], size[3]), True)(out)  # do not support GPU yet
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class _MPUnit(nn.Cell):
    def __init__(self, in_channels, out_channles, atrous_rates, activation='relu'):
        super(_MPUnit, self).__init__()
        self.mpu0 = nn.Conv2dBnAct(in_channels, out_channles, 1,
                                   weight_init='xavier_uniform',
                                   has_bn=True,
                                   activation=activation)
        self.mpu1 = nn.Conv2dBnAct(in_channels, out_channles, 3, 1, 'pad',
                                   padding=atrous_rates[0],
                                   dilation=atrous_rates[0],
                                   group=in_channels,
                                   weight_init='xavier_uniform',
                                   has_bn=True,
                                   activation=activation)
        self.mpu2 = nn.Conv2dBnAct(in_channels, out_channles, 3, 1, 'pad',
                                   padding=atrous_rates[1],
                                   dilation=atrous_rates[1],
                                   group=in_channels,
                                   weight_init='xavier_uniform',
                                   has_bn=True,
                                   activation=activation)
        self.mpu3 = nn.Conv2dBnAct(in_channels, out_channles, 3, 1, 'pad',
                                   padding=atrous_rates[2],
                                   dilation=atrous_rates[2],
                                   group=in_channels,
                                   weight_init='xavier_uniform',
                                   has_bn=True,
                                   activation=activation)
        self.proj = nn.Conv2dBnAct(4 * out_channles, out_channles, 1,
                                   weight_init='xavier_uniform',
                                   has_bn=True,
                                   activation=activation)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        x0 = self.mpu0(x)
        x1 = self.mpu1(x)
        x2 = self.mpu2(x)
        x3 = self.mpu3(x)
        x = self.concat((x0, x1, x2, x3))
        x = self.proj(x)
        return x


class _EPRModule(nn.Cell):
    def __init__(self, in_channels, out_channels, atrous_rates, activation='relu', down_sample=False):
        super(_EPRModule, self).__init__()
        stride = 2 if down_sample else 1

        self.pyramid = _MPUnit(in_channels, out_channels, atrous_rates, activation)
        self.compact = nn.Conv2dBnAct(out_channels, out_channels, 3, stride, 'pad', 1,
                                      weight_init='xavier_uniform',
                                      has_bn=True,
                                      activation=activation)
        if (out_channels != in_channels) or down_sample:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, weight_init='xavier_uniform')
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        self.act = nn.ReLU()
        self.add = P.TensorAdd()

    def construct(self, x):
        identity = x
        out = self.pyramid(x)
        out = self.compact(out)
        if self.skip is not None:
            identity = self.skip(identity)
            identity = self.skip_bn(identity)
        out = self.add(out, identity)
        return self.act(out)


def get_eprnet(**kwargs):
    return EPRNet(**kwargs)
