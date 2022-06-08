# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

'''agw net define '''

from cmath import inf
import os
import warnings

from collections import OrderedDict

import mindspore
from mindspore import Tensor, nn
import mindspore.ops as P
from mindspore import load_checkpoint, load_param_into_net
import mindspore.common.initializer as init

from resnet import resnet50, resnet50nl
from utils.config import config


class GeneralizedMeanPooling(nn.Cell):
    '''
    implementation of GEM Pooling in AGW
    '''

    def __init__(self, norm, trainable_norm=True, output_size=1, eps=1e-6):
        super().__init__()
        assert norm > 0

        if trainable_norm:
            self.p = mindspore.Parameter(Tensor(1, dtype=mindspore.float32))
        else:
            self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

        self.pow = P.Pow()

    def construct(self, x):
        x = P.clip_by_value(x, clip_value_min=self.eps, clip_value_max=inf)
        x = self.pow(x, self.p)
        x = nn.AvgPool2d(kernel_size=x.shape[2:], stride=1)(x)
        x = self.pow(x, 1.0/self.p)
        return x


class AvgPool2d(nn.Cell):
    '''
    wrapped AvgPool2d
    '''

    def construct(self, x):
        x = nn.AvgPool2d(kernel_size=x.shape[2:], stride=1)(x)
        return x


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
        m.gamma.set_data(init.initializer(init.Normal(
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


class AGW(nn.Cell):
    '''
    definition of the AGW net
    '''

    def __init__(self, num_classes, last_conv_stride=1):
        super().__init__()
        self.in_planes = 2048
        self.num_classes = num_classes

        if config.non_local:
            self.backbone = resnet50nl(
                pretrain=config.backbone_pretrain_dir, last_conv_stride=last_conv_stride)
        else:
            self.backbone = resnet50(
                pretrain=config.backbone_pretrain_dir, last_conv_stride=last_conv_stride)

        if config.gem_pool:
            self.global_pool = GeneralizedMeanPooling(1)
        else:
            self.global_pool = AvgPool2d()

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.beta.requires_grad = False
        weights_init_kaiming(self.bottleneck)
        self.classifier = nn.Dense(
            self.in_planes, self.num_classes, has_bias=False)

        weights_init_classifier(self.classifier)

    def construct(self, x):
        '''
        forward
        '''
        x = self.backbone(x)
        global_feat = self.global_pool(x)

        global_feat = global_feat.view(
            (global_feat.shape[0], -1))  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if not self.training:
            return global_feat

        cls_score = self.classifier(feat)
        return cls_score, global_feat


def init_pretrained_weights(model, pretrained_param_dir):
    """
    Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """

    filename = 'init_agw.ckpt'
    file = os.path.join(pretrained_param_dir, filename)
    print(file)
    if not os.path.exists(file):
        raise ValueError(
            'The file:{} does not exist.'.format(file)
        )
    param_dict = load_checkpoint(file)
    model_dict = model.parameters_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in param_dict.items():
        if k in model_dict and model_dict[k].data.shape == v.shape:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    load_param_into_net(model, model_dict)

    if matched_layers:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
            format(file)
        )
        if discarded_layers:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


def create_agw_net(num_class=1500, last_stride=1, pretrained=False, pretrained_dir='./model_utils'):

    net = AGW(num_class, last_stride)
    if pretrained:
        init_pretrained_weights(net, pretrained_dir)
    return net

# =============================================


class AGWLoss(nn.LossBase):
    '''
    wrapped Loss, passed to Model
    '''

    def __init__(self, ce=None, tri=None):
        super().__init__()
        self.ce = ce
        self.tri = tri

    def construct(self, logits, labels):
        '''
        forward
        '''
        cls_score, global_feat = logits
        loss = self.ce(cls_score, labels)
        if self.tri:
            tri_loss = self.tri(global_feat, labels)
            loss += tri_loss
        return loss
