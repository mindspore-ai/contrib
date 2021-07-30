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
"""vib"""

import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeNormal, Zero, Normal

from model_main import NormalWithMean


def weights_init_kaiming(m):
    """weights_init_kaiming"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_data(initializer(HeNormal(negative_slope=0, mode='fan_in'), m.weight.shape, m.weight.dtype))
    elif classname.find('Linear') != -1:
        m.weight.set_data(initializer(HeNormal(negative_slope=0, mode='fan_out'), m.weight.shape, m.weight.dtype))
        m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        m.gamma.set_data(initializer(NormalWithMean(mu=1, sigma=0.01), m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(initializer(Zero(), m.beta.shape, m.beta.dtype))


def weights_init_classifier(m):
    """weights_init_classifier"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.gamma.set_data(initializer(Normal(sigma=0.001), m.gamma.shape, m.gamma.dtype))
        if m.bias:
            m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))


########################################################################
# Variational Distillation
########################################################################
class ChannelCompress(nn.Cell):
    """ChannelCompress"""
    def __init__(self, in_ch=2048, out_ch=256):
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Dense(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_features=num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Dense(num_bottleneck, 500)]
        add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.ReLU()]

        add_block += [nn.Dense(500, out_ch)]

        add_block = nn.SequentialCell(add_block)

        weights_init_kaiming(add_block)

        self.model = add_block

    def construct(self, x):
        x = self.model(x)
        return x


########################################################################
# Variational Distillation
########################################################################
class VIB(nn.Cell):
    """VIB"""
    def __init__(self, in_ch=2048, z_dim=256, num_class=395):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB
        classifier = []
        classifier += [nn.Dense(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Dense(self.out_ch // 2, self.num_class)]
        classifier = nn.SequentialCell(classifier)
        weights_init_classifier(classifier)
        self.classifier = classifier

    def construct(self, v):
        z_given_v = self.bottleneck(v)
        logits_given_z = self.classifier(z_given_v)
        return z_given_v, logits_given_z
