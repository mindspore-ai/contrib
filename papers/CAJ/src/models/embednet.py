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
"""model_main.py"""
import mindspore.common.initializer as init
import mindspore.ops as P

from mindspore import nn
from mindspore.common.initializer import Normal

from src.models.resnet import resnet50, resnet50_share, resnet50_specific


def weights_init_kaiming(m):
    """
    function of weights_init_kaiming
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_data(init.initializer(init.HeNormal(
            negative_slope=0, mode='fan_in'), m.weight.shape, m.weight.dtype))
    elif classname.find('Linear') != -1:
        m.weight.set_data(init.initializer(init.HeNormal(
            negative_slope=0, mode='fan_out'), m.weight.shape, m.weight.dtype))
        m.bias.set_data(init.initializer(
            init.Zero(), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        m.gamma.set_data(init.initializer(Normal(
            mean=1.0, sigma=0.01), m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(init.initializer(
            init.Zero(), m.beta.shape, m.beta.dtype))


def weights_init_classifier(m):
    """
    function of weights_init_classifier
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.gamma.set_data(init.initializer(init.Normal(
            sigma=0.001), m.gamma.shape, m.gamma.dtype))
        if m.bias:
            m.bias.set_data(init.initializer(
                init.Zero(), m.bias.shape, m.bias.dtype))


class Normalize(nn.Cell):
    """
    class of normalize
    """
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


class Visible(nn.Cell):
    """
    class of visible module
    """
    def __init__(self, pretrain=""):
        super(Visible, self).__init__()
        self.visible = resnet50_specific(pretrain=pretrain)

    def construct(self, x):

        x = self.visible(x)
        return x


class Thermal(nn.Cell):
    """
    class of thermal module
    """
    def __init__(self, pretrain=""):
        super(Thermal, self).__init__()

        self.thermal = resnet50_specific(pretrain=pretrain)

    def construct(self, x):

        x = self.thermal(x)

        return x


class BASE(nn.Cell):
    def __init__(self, pretrain=""):
        super(BASE, self).__init__()
        self.base = resnet50_share(pretrain=pretrain)

    def construct(self, x):
        x = self.base(x)
        return x


class ResNet50(nn.Cell):
    def __init__(self, pretrain=""):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrain=pretrain)

    def construct(self, x):
        x = self.resnet(x)
        return x


class EmbedNet(nn.Cell):
    """
    Backbone net
    """
    def __init__(self, pretrain="", class_num=395, drop=0.2, part=0, alpha=0.2, nheads=4, low_dim=0):
        super(EmbedNet, self).__init__()
        self.thermal_module = ThermalModule(pretrain=pretrain)
        self.visible_module = VisibleModule(pretrain=pretrain) # changed here
        self.base_resnet = BaseResnet(pretrain=pretrain)
        self.non_local = "off"
        # self.non_local_net = None
        if self.non_local == "on":
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.nonlocal_1 = nn.CellList(
                [NonLocal(256) for i in range(non_layers[0])])
            self.nonlocal_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.nonlocal_2 = nn.CellList(
                [NonLocal(512) for i in range(non_layers[1])])
            self.nonlocal_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.nonlocal_3 = nn.CellList(
                [NonLocal(1024) for i in range(non_layers[2])])
            self.nonlocal_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.nonlocal_4 = nn.CellList(
                [NonLocal(2048) for i in range(non_layers[3])])
            self.nonlocal_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])
            self.nonlocal_1_idx = [-1]
            self.nonlocal_4_idx = [-1]
            # self.non_local_net = non_local_net(self)

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.requires_grad = False
        self.classifier = nn.Dense(2048, class_num, has_bias=False)
        weights_init_kaiming(self.bottleneck)
        weights_init_classifier(self.classifier)
        adaptive_avg_pool_2d = P.AdaptiveAvgPool2D((1, 2))
        self.avgpool = adaptive_avg_pool_2d
        self.gm_pool = "on"
        self.dropout = drop
        self.part = part
        self.class_num = class_num
        self.nheads = nheads
        self.alpha = alpha
        self.low_dim = low_dim
        self.len1 = len(self.base_resnet.base.layer1)
        self.len2 = len(self.base_resnet.base.layer2)
        self.len3 = len(self.base_resnet.base.layer3)
        self.len4 = len(self.base_resnet.base.layer4)
        self.list1 = []
        self.list2 = []
        self.list3 = []
        self.list4 = []
        for item in self.base_resnet.base.layer1:
            self.list1.append(item)
        for item in self.base_resnet.base.layer2:
            self.list2.append(item)
        for item in self.base_resnet.base.layer3:
            self.list3.append(item)
        for item in self.base_resnet.base.layer4:
            self.list4.append(item)

    def non_local_out(self, x):
        """
        non local output: used only non_local == "on"
        """
        nonlocal1_counter = 0
        for i in range(self.len1):
            x = self.list1[i](x)
            if i == self.nonlocal_1_idx[nonlocal1_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_1[nonlocal1_counter](x)
                nonlocal1_counter += 1
        # Layer 2
        nonlocal2_counter = 0
        for i in range(self.len2):
            x = self.list2[i](x)
            if i == self.nonlocal_2_idx[nonlocal2_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_2[nonlocal2_counter](x)
                nonlocal2_counter += 1
        # Layer 3
        nonlocal3_counter = 0
        for i in range(self.len3):
            x = self.list3[i](x)
            if i == self.nonlocal_3_idx[nonlocal3_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_3[nonlocal3_counter](x)
                nonlocal3_counter += 1
        # Layer 4
        nonlocal4_counter = 0
        for i in range(self.len4):
            x = self.list4[i](x)
            if i == self.nonlocal_4_idx[nonlocal4_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_4[nonlocal4_counter](x)
                nonlocal4_counter += 1
        return x

    def construct(self, x1, x2, modal=0):
        """
        x1: img1
        x2: img2
        """
        concat = P.Concat()
        x = None
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = concat((x1, x2))
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        if self.non_local == "on":
            x = self.non_local_out(x)
        else:
            x = self.base_resnet(x)

        if self.gm_pool == 'on':
            op = P.ReduceMean(keep_dims=False)
            b, c, _, _ = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            # x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
            x_pool = (op(x**p, -1) + 1e-12) ** (1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.shape[0], x_pool.shape[1])
        feat = self.bottleneck(x_pool)
        if self.training:
            return x_pool, x_pool, self.classifier(feat), self.classifier(feat)
        return self.l2norm(x_pool), self.l2norm(feat)


class VisibleModule(nn.Cell):
    """
    class of visible module
    """
    def __init__(self, pretrain):
        super(VisibleModule, self).__init__()
        model_v = resnet50(pretrain=pretrain, last_conv_stride=1)
        self.visible = model_v
    def construct(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

class ThermalModule(nn.Cell):
    """
    class of thermal module
    """
    def __init__(self, pretrain):
        super(ThermalModule, self).__init__()
        model_v = resnet50(pretrain=pretrain, last_conv_stride=1)
        self.thermal = model_v
    def construct(self, x):
        """
        x: input (img)
        """
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class BaseResnet(nn.Cell):
    """
    class of base resnet
    """
    def __init__(self, pretrain):
        super(BaseResnet, self).__init__()
        model_base = resnet50(pretrain=pretrain, last_conv_stride=1)
        self.base = model_base
    def construct(self, x):
        """
        x: input (img)
        """
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class NonLocal(nn.Cell):
    """
    class of non local
    """
    def __init__(self, in_channels, reduc_ratio=2):
        super(NonLocal, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio
        self.g = nn.SequentialCell(
            [
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, \
                kernel_size=1, stride=1, padding=0)
            ]
        )
        self.weight = nn.SequentialCell(
            [
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, \
                kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels, gamma_init='zeros', beta_init='zeros')
            ]
        )
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, \
        kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, \
        kernel_size=1, stride=1, padding=0)

    def construct(self, x):
        """
        x: input(int)
        """
        trans = P.Transpose()
        batch_size = x.shape[0]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = trans(g_x, (0, 2, 1))

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = trans(theta_x, (0, 2, 1))
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # mat_net = nn.MatMul()
        # matmul = P.MatMul()
        f = P.matmul(theta_x, phi_x)
        n_shape = f.shape[-1]
        f_div_c = f / n_shape

        y = P.matmul(f_div_c, g_x)
        y = trans(y, (0, 2, 1))
        y = y.view(batch_size, self.inter_channels, *x.shape[2:])
        wy = self.weight(y)
        z = wy + x

        return z
