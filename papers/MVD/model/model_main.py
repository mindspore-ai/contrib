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
"""model_main"""

import os
import numpy as np
import psutil

import mindspore as ms
import mindspore.ops as P
import mindspore.common.initializer as init
from mindspore import nn

from mindspore.common.initializer import Initializer, _assignment, random_normal

from model.resnet import resnet50
from model.vib import VIB


def show_memory_info(hint=""):
    """show_memory_info"""
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")


def to_edge(x):
    """to_edge"""
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    xx = 0.2989 * r + 0.5870 * g + 0.1440 * b
    xx = xx.view((xx.shape[0], 1, xx.shape[1], xx.shape[2]))
    return xx  # N x 1 x h x w


class NormalWithMean(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, normal array.
    """

    def __init__(self, mu=0, sigma=0.01):
        super(NormalWithMean, self).__init__(sigma=sigma)
        self.mu = mu
        self.sigma = sigma

    def _initialize(self, arr):
        seed, seed2 = self.seed
        output_tensor = ms.Tensor(
            np.zeros(arr.shape, dtype=np.float32) + np.ones(arr.shape, dtype=np.float32) * self.mu)
        random_normal(arr.shape, seed, seed2, output_tensor)
        output_data = output_tensor.asnumpy()
        output_data *= self.sigma
        _assignment(arr, output_data)


def weights_init_kaiming(m):
    """weights_init_kaiming"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_data(
            init.initializer(init.HeNormal(negative_slope=0, mode='fan_in'), m.weight.shape, m.weight.dtype))
    elif classname.find('Linear') != -1:
        m.weight.set_data(
            init.initializer(init.HeNormal(negative_slope=0, mode='fan_out'), m.weight.shape, m.weight.dtype))
        m.bias.set_data(init.initializer(init.Zero(), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        m.gamma.set_data(init.initializer(NormalWithMean(mu=1, sigma=0.01), m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(init.initializer(init.Zero(), m.beta.shape, m.beta.dtype))


def weights_init_classifier(m):
    """weights_init_classifier"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.gamma.set_data(init.initializer(init.Normal(sigma=0.001), m.gamma.shape, m.gamma.dtype))
        if m.bias:
            m.bias.set_data(init.initializer(init.Zero(), m.bias.shape, m.bias.dtype))


class Normalize(nn.Cell):
    """Normalize"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        self.pow = P.Pow()
        self.sum = P.ReduceSum(keep_dims=True)
        self.div = P.Div()

    def construct(self, x):
        norm = self.pow(x, self.power)
        norm = self.sum(norm, 1)
        norm = self.pow(norm, 1. / self.power)
        out = self.div(x, norm)
        return out


class VisibleBackbone(nn.Cell):
    """VisibleBackbone"""
    def __init__(self, num_class=395, arch="resnet50", pretrain=""):
        super(VisibleBackbone, self).__init__()

        self.visible = resnet50(num_class=num_class, pretrain=pretrain)
        self.arch = arch

    def construct(self, x):
        x = self.visible(x)

        return x


class ThermalBackbone(nn.Cell):
    """ThermalBackbone"""
    def __init__(self, num_class=395, arch="resnet50", pretrain=""):
        super(ThermalBackbone, self).__init__()

        self.thermal = resnet50(num_class=num_class, pretrain=pretrain)
        self.arch = arch

    def construct(self, x):
        x = self.thermal(x)

        return x


class SharedBackbone(nn.Cell):
    """SharedBackbone"""
    def __init__(self, num_class=395, arch="resnet50", pretrain=""):
        super(SharedBackbone, self).__init__()

        self.base = resnet50(num_class=num_class, pretrain=pretrain)
        self.arch = arch

    def construct(self, x):
        x = self.base(x)

        return x


class EmbedNet(nn.Cell):
    """EmbedNet"""
    def __init__(self, num_class=395, drop=0.2, z_dim=512, arch="resnet50", pretrain=""):
        super(EmbedNet, self).__init__()

        self.rgb_backbone = VisibleBackbone(num_class=num_class, arch=arch, pretrain=pretrain)
        self.ir_backbone = ThermalBackbone(num_class=num_class, arch=arch, pretrain=pretrain)
        self.shared_backbone = SharedBackbone(num_class=num_class, arch=arch, pretrain=pretrain)

        pool_dim = 2048
        self.rgb_bottleneck = VIB(in_ch=pool_dim, z_dim=z_dim, num_class=num_class)
        self.ir_bottleneck = VIB(in_ch=pool_dim, z_dim=z_dim, num_class=num_class)
        self.shared_bottleneck = VIB(in_ch=pool_dim, z_dim=z_dim, num_class=num_class)

        self.dropout = drop

        self.l2norm = Normalize(2)

        self.avgpool = P.ReduceMean(keep_dims=True)
        self.cat = P.Concat()
        self.cat_dim1 = P.Concat(axis=1)

    def construct(self, x1, x2=None, mode=0):
        """build"""
        # visible branch
        if mode == 0:
            x = self.cat((x1, x2))
        else:
            x = x1
        # backbone 输出为二元组(feature, logits),下同
        v_observation = self.rgb_backbone(x)
        v_representation = self.rgb_bottleneck(v_observation[0])

        # infarred branch
        x_grey = to_edge(x)
        i_ms_input = self.cat_dim1([x_grey, x_grey, x_grey])

        i_observation = self.ir_backbone(i_ms_input)
        i_representation = self.ir_bottleneck(i_observation[0])

        # modal shared branch
        v_ms_observation = self.shared_backbone(x)
        v_ms_representation = self.shared_bottleneck(v_ms_observation[0])

        i_ms_observation = self.shared_backbone(i_ms_input)
        i_ms_representation = self.shared_bottleneck(i_ms_observation[0])

        if self.training:
            return v_observation, v_representation, v_ms_observation, v_ms_representation, \
                   i_observation, i_representation, i_ms_observation, i_ms_representation

        v_observation = self.l2norm(v_observation[0])
        v_representation = self.l2norm(v_representation[0])
        v_ms_observation = self.l2norm(v_ms_observation[0])
        v_ms_representation = self.l2norm(v_ms_representation[0])

        i_observation = self.l2norm(i_observation[0])
        i_representation = self.l2norm(i_representation[0])
        i_ms_observation = self.l2norm(i_ms_observation[0])
        i_ms_representation = self.l2norm(i_ms_representation[0])

        feat_v = self.cat_dim1((v_observation, v_representation))
        feat_i = self.cat_dim1((i_observation, i_representation))
        feat_v_shared = self.cat_dim1((v_ms_observation, v_ms_representation))
        feat_i_shared = self.cat_dim1((i_ms_observation, i_ms_representation))

        return feat_v, feat_v_shared, feat_i, feat_i_shared
