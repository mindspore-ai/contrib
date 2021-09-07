# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""cell_ops"""

from mindspore import nn
from mindspore.ops import Mul, Pad, Concat
from mindspore.ops import operations as P

# OPS defines operations for micro cell structures
OPS = {
    '0': lambda c_in, c_out, stride, affine, track_running_stats: Zero(c_in, c_out, stride),
    '1': lambda c_in, c_out, stride, affine, track_running_stats: Identity() if (
        stride == 1 and c_in == c_out) else FactorizedReduce(c_in, c_out, stride, affine, track_running_stats),
    '2': lambda c_in, c_out, stride, affine, track_running_stats: ReLUConvBN(c_in, c_out, 1, stride, 0,
                                                                             1, affine, track_running_stats),
    '3': lambda c_in, c_out, stride, affine, track_running_stats: ReLUConvBN(c_in, c_out, 3, stride, 1,
                                                                             1, affine, track_running_stats)
}


class ReLUConvBN(nn.Cell):
    """ReLUConvBN"""

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine, track_running_stats,
                 activation='relu'):
        super(ReLUConvBN, self).__init__()
        if activation == 'leaky':
            ops = [nn.LeakyReLU()]
        elif activation == 'relu':
            ops = [nn.ReLU()]
        else:
            raise ValueError(f"invalid activation {activation}")

        ops += [nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                          pad_mode="pad", has_bias=False),
                nn.BatchNorm2d(c_out, affine=affine, use_batch_statistics=track_running_stats)]
        self.ops = nn.SequentialCell(ops)
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride

    def construct(self, x):
        return self.ops(x)

    def extra_repr(self):
        """extra_repr"""
        return 'c_in={c_in}, c_out={c_out}, stride={stride}'.format(**self.__dict__)


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        return x


class Zero(nn.Cell):
    """Zero"""

    def __init__(self, c_in, c_out, stride):
        super(Zero, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.is_zero = True
        self.mul = Mul()
        self.zeros = P.Zeros()

    def construct(self, x):
        """construct"""
        if self.c_in == self.c_out:
            if self.stride == 1:
                return self.mul(x, 0.)
            return self.mul(x[:, :, ::self.stride, ::self.stride], 0.)

        shape = list(x.shape)
        shape[1], shape[2], shape[3] = self.c_out, (shape[2] + 1) // self.stride, (shape[3] + 1) // self.stride
        zeros = self.zeros(x.shape, x.dtype)
        return zeros

    def extra_repr(self):
        """extra_repr"""
        return 'c_in={c_in}, c_out={c_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Cell):
    """FactorizedReduce"""

    def __init__(self, c_in, c_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.c_in = c_in
        self.c_out = c_out
        self.relu = nn.ReLU()
        if c_out % 2 == 0:
            raise 'c_out : {:}'.format(c_out)
        c_outs = [c_out // 2, c_out - c_out // 2]
        self.convs = nn.CellList()
        for i in range(2):
            self.convs.append(nn.Conv2d(c_in, c_outs[i], 1, stride=stride, pad_mode="valid", has_bias=False))
        self.pad = Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.bn = nn.BatchNorm2d(c_out, affine=affine, use_batch_statistics=track_running_stats)
        self.cat = Concat(axix=1)

    def construct(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = self.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])])
        # print(self.convs[0](x).shape, self.convs[1](y[:,:,1:,1:]).shape)
        # print(out.shape)

        out = self.bn(out)
        return out

    def extra_repr(self):
        """extra_repr"""
        return 'c_in={c_in}, c_out={c_out}, stride={stride}'.format(**self.__dict__)


class ConvLayer(nn.Cell):
    """ConvLayer"""

    def __init__(self, in_channel, out_channel, kernel, stride, padding, activation, norm):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, pad_mode="pad")
        self.activation = activation
        if norm:
            if norm == nn.BatchNorm2d:
                self.norm = norm(out_channel)
            else:
                self.norm = norm
                self.conv = norm(self.conv)
        else:
            self.norm = None

    def construct(self, x):
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeconvLayer(nn.Cell):
    """DeconvLayer"""

    def __init__(self, in_channel, out_channel, kernel, stride, padding, activation, norm):
        super(DeconvLayer, self).__init__()
        self.pad = Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.conv = nn.Conv2dTranspose(in_channel, out_channel, kernel, stride=stride, padding=padding,
                                       pad_mode="pad")
        self.activation = activation
        if norm == nn.BatchNorm2d:
            self.norm = norm(out_channel)
        else:
            self.norm = norm

    def construct(self, x):
        x = self.conv(x)
        x = self.pad(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
