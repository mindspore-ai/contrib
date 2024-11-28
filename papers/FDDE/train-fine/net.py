# Copyright 2022 Huawei Technologies Co., Ltd
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
"""net"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net
from resnet.resnet50 import resnet50


class MM(nn.Cell):
    """module"""

    def __init__(self):
        super(MM, self).__init__()
        # relu = nn.ReLU()
        self.conv1 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, pad_mode="pad"),
             nn.BatchNorm2d(64),
             nn.ReLU()])
        self.conv2 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, pad_mode="pad"),
             nn.BatchNorm2d(64),
             nn.ReLU()])
        self.conv3 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])
        self.conv4 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=4, padding=4, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])
        self.conv5 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=8, padding=8, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])

    def construct(self, out1, out2):
        """     construct     """
        resize_bilinear = nn.ResizeBilinear()
        avgpool = nn.AvgPool2d(kernel_size=352 // 4)
        x = self.conv1(out1 + resize_bilinear(out2, size=out1.shape[2:]))
        out = avgpool(x)

        out = out * x

        out = self.conv2(out) + self.conv3(out) + self.conv4(out) + self.conv5(out)
        return out


class NN(nn.Cell):
    """module"""

    def __init__(self):
        super(NN, self).__init__()
        self.conv_l2 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])
        self.conv_l3 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])
        self.conv_l4 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=6, dilation=6, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])
        self.conv_l5 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8, pad_mode="pad"),
             nn.BatchNorm2d(64), nn.ReLU()])

    def construct(self, in_c):
        """     construct     """
        resize_bilinear = nn.ResizeBilinear()
        out5 = self.conv_l5(in_c[0])
        out4 = self.conv_l4(
            in_c[1] + resize_bilinear(out5, size=in_c[1].shape[2:], align_corners=True))
        out3 = self.conv_l3(
            in_c[2] + resize_bilinear(out4, size=in_c[2].shape[2:], align_corners=True))
        out2 = self.conv_l2(
            in_c[3] + resize_bilinear(out3, size=in_c[3].shape[2:], align_corners=True))

        return (out2, out3, out4, out5)


class WW(nn.Cell):
    """module"""

    def __init__(self):
        super(WW, self).__init__()
        self.conv12 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, pad_mode="pad"),
             nn.BatchNorm2d(64),
             nn.ReLU()])
        self.conv13 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, pad_mode="pad"),
             nn.BatchNorm2d(64),
             nn.ReLU()])
        self.conv14 = nn.SequentialCell(
            [nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, pad_mode="pad"),
             nn.BatchNorm2d(64),
             nn.ReLU()])

    def construct(self, edg, in_c):
        """     construct     """
        resize_bilinear = nn.ResizeBilinear()
        out12 = self.conv12(edg + resize_bilinear(in_c[0], size=edg.shape[2:], align_corners=True))
        out13 = self.conv13(
            edg + resize_bilinear(in_c[1], size=edg.shape[2:], align_corners=True) + out12)
        out14 = self.conv14(
            edg + resize_bilinear(in_c[2], size=edg.shape[2:], align_corners=True) + out13)

        return (out12, out13, out14)


class FDDE(nn.Cell):
    """module"""

    def __init__(self, cfg):
        super(FDDE, self).__init__()
        self.cfg = cfg
        net = resnet50()
        # 加载预训练模型
        param_dict = load_checkpoint('resnet/resnet50.ckpt')
        # 给网络加载参数
        load_param_into_net(net, param_dict)

        self.bkbone = net
        self.conv5 = nn.SequentialCell(nn.Conv2d(2048, 64, kernel_size=1, pad_mode="valid"),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.SequentialCell(nn.Conv2d(1024, 64, kernel_size=1, pad_mode="valid"),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv3 = nn.SequentialCell(nn.Conv2d(512, 64, kernel_size=1, pad_mode="valid"),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.SequentialCell(nn.Conv2d(256, 64, kernel_size=1, pad_mode="valid"),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.conv1 = nn.SequentialCell(nn.Conv2d(64, 64, kernel_size=1, pad_mode="valid"),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode="pad"),
                                       nn.BatchNorm2d(64), nn.ReLU())

        self.convf = nn.SequentialCell(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, pad_mode="valid"))
        self.conv_edg = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, pad_mode="valid")

        self.nn = NN()
        self.mm = MM()
        self.ww = WW()

    def construct(self, x, shape=None):
        """methods"""
        resize_bilinear = nn.ResizeBilinear()
        conc = ops.Concat(axis=1)
        if shape is None:
            shape = x.shape[2:]
        out1, out2, out3, out4, out5 = self.bkbone(x)
        out1, out2, out3, out4, out5 = self.conv1(out1), self.conv2(out2), self.conv3(
            out3), self.conv4(out4), self.conv5(out5)

        (out_l2, out_l3, out_l4, out_l5) = self.nn([out5, out4, out3, out2])

        out_l1 = self.mm(out1, out_l2)
        (_, _, out14) = self.ww(edg=out_l1, in_c=[out_l2, out_l3, out_l4])

        out14_5 = out14 + resize_bilinear(out_l5, size=out14.shape[2:], align_corners=True)

        out = self.convf(conc((out_l1, out14_5)))

        edg = self.conv_edg(out_l1)
        edg = resize_bilinear(edg, size=shape, align_corners=True)
        out = resize_bilinear(out, size=shape, align_corners=True)
        return edg, out
