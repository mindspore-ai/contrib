# import torch.nn as nn
import sys
sys.path.append('.')

import numpy
import math
# import torch.utils.model_zoo as model_zoo
# import torch
# from DCNv2.dcn_v2 import DCN
# BatchNorm2d = nn.BatchNorm2d
from mindspore.ops import Concat
import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor
from scipy.stats import truncnorm
BatchNorm2d = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def maxpool4mindspore(input):
    shape1 = list(input.shape)
    shape2 = list(input.shape)
    shape1[2] = 1
    shape2[2] = shape2[2] + 2
    shape2[3] = 1
    shape1 = tuple(shape1)
    shape2 = tuple(shape2)
    inf1 = Tensor(np.array(np.ones(shape1)), mstype.float32) * -np.inf
    inf2 = Tensor(np.array(np.ones(shape2)), mstype.float32) * -np.inf

    concat_horizon = Concat(2)
    concat1 = concat_horizon((inf1,input,inf1))

    concat_vertical = Concat(3)
    concat2 = concat_vertical((inf2,concat1,inf2))

    maxpool = nn.MaxPool2d(kernel_size=3, stride=2,  pad_mode='valid')
    result = maxpool(concat2)

    return result
    

def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, pad_mode='pad', weight_init='normal', has_bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        # self.conv2 = conv3x3(planes, planes)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   padding=1, pad_mode='pad', has_bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from assets.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from assets.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1,
                pad_mode='pad',
                has_bias=True)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                deformable_groups=deformable_groups,
                has_bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, pad_mode='pad', has_bias=False)
        self.bn1 = BatchNorm2d(planes)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, pad_mode='pad', has_bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                from assets.ops.dcn import DeformConv
                conv_op = DeformConv
                offset_channels = 18
            else:
                from assets.ops.dcn import ModulatedDeformConv
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes, deformable_groups * offset_channels,
                kernel_size=3,
                padding=1,
                pad_mode='pad',
                has_bias=True)
            self.conv2 = conv_op(
                planes, planes, kernel_size=3, padding=1, stride=stride,
                deformable_groups=deformable_groups, has_bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, padding=0, pad_mode='pad', has_bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, -9:, :, :].sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self, block, layers, num_classes=1000, 
                 dcn=None, stage_with_dcn=(False, False, False, False)):
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad',
                               has_bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        self.avgpool = nn.AvgPool2d(7, stride=1, pad_mode='same')
        self.fc = nn.Dense(512 * block.expansion, num_classes)
    
        self.smooth = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=1, pad_mode='pad', has_bias=True)    

        for m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.cells_and_names():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = maxpool4mindspore(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x, x2, x3, x4, x5


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet18']), strict=False)
    return model

def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                    dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                    stage_with_dcn=[False, True, True, True], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet50']), strict=False)
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    
    replace_list = ['layer2','layer3','layer4']
    for name,module in model.named_children():
        if name not in replace_list:
            continue
        replace(module)

    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet50']), strict=False)
    return model


def replace(layers):
    for name,module in layers.named_children():
        if isinstance(module,nn.Conv2d):
            if 'conv2' in name:
                print('DCN')
        else:
            replace(module)


'''
def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   dcn=dict(modulated=True,
                            deformable_groups=1,
                            fallback_on_stride=False),
                   stage_with_dcn=[False, True, True, True],
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=False)
    return model
'''

def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(
    #         model_urls['resnet152']), strict=False)
    return model


if __name__ == '__main__':
    # import torch
    import numpy
    input = Tensor(numpy.random.randn(4, 3, 512, 512), mstype.float32)

    print(input.shape)
    #  

    # input = torch.randn((4, 3, 512, 512))
    net = resnet50()
    # net = deformable_resnet50()

    C1, C2, C3, C4, C5 = net(input)
    print(C1.shape)
    print(C2.shape)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)



    # import torch
    # t1 = torch.tensor([[1,1,1]])
    # t2 = torch.tensor([[2,2,2]])
    # t3 = torch.tensor([[3,3,3]])

    # t = torch.cat((t1,t2,t3),dim=0)
    # print(t)

    # t1 = Tensor(np.array([[1,1,1]]),mstype.int32)
    # t2 = Tensor(np.array([[2,2,2]]),mstype.int32)
    # t3 = Tensor(np.array([[3,3,3]]),mstype.int32)

    # t = np.concatenate((t1,t2,t3),axis=0)
    # print(t)

    # import mindspore.ops as ops
    # import torch
    # import torch.nn.functional as F
    # # input_tensor = torch.tensor([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]])
    # input_tensor = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), mstype.float32)
    # resize = ops.ResizeNearestNeighbor((4, 6))
    # # x = F.interpolate(input_tensor, scale_factor=2)
    # output = resize(input_tensor)
    # print(output)

    # import torch
    # import numpy as np
    # x = torch.zeros((2,1))
    # print(x.shape)
    # y = np.zeros((2,1))
    # y = Tensor(y, mstype.float32)
    # print(y.shape)
    

