import sys
sys.path.append('.')

import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.ops as ops
from mindspore import Tensor, context

from network.resnet import resnet50
from mindspore import load_checkpoint, load_param_into_net



class FPN_Block(nn.Cell):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode='pad', has_bias=True)
        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)
        self.leaky_relu = nn.LeakyReLU(1e-02)

    def construct(self, x):
        x = self.conv3x3_1(x)
        x = self.leaky_relu(x)
        x = self.conv1x1_1(x)
        x = self.leaky_relu(x)
        x = self.conv3x3_2(x)
        return x


class FPN(nn.Cell):

    def __init__(self, dcn=False, is_training=True):
        super().__init__()
        self.dcn = dcn
        self.is_training = is_training

        if not self.dcn:
            print('resnet50')
            self.backbone = resnet50()
        else:
            print('resnet50 with DCN')
            # self.backbone = deformable_resnet50()
            
        self.merge5 = FPN_Block(2048, 128)
        self.merge4 = FPN_Block(1024 + 128, 128)
        self.merge3 = FPN_Block(512 + 128, 128)

    def construct(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        P5 = self.merge5(C5)

        # x = F.interpolate(P5, scale_factor=2)
        upsample_P5 = ops.ResizeNearestNeighbor((2 * P5.shape[2], 2 * P5.shape[3]))
        x = upsample_P5(P5)
        x = np.concatenate((C4, x), axis=1)
        # x = Tensor(x, mstype.float32)
        P4 = self.merge4(x)

        # x = F.interpolate(P4, scale_factor=2)
        upsample_P4 = ops.ResizeNearestNeighbor((2 * P4.shape[2], 2 * P4.shape[3]))
        x = upsample_P4(P4)
        x = np.concatenate((C3, x), axis=1)
        # x = Tensor(x, mstype.float32)
        P3 = self.merge3(x)

        return P3, P4, P5


class TextNet(nn.Cell):

    def __init__(self, k=5, dcn=False, is_training=True):
        super().__init__()
        self.k = k
        self.is_training = is_training
        self.dcn = dcn
        self.fpn = FPN(self.dcn, self.is_training)

        self.in_channel = 128

        # ##class and regression branch
        self.cls_out = 4 # +1
        self.cls_predict = nn.SequentialCell([
            nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            nn.Conv2d(32, self.cls_out, kernel_size=1, stride=1, padding=0, pad_mode='pad', has_bias=True)
        ])
        # ##class and regression branch 
        self.reg_out = (2 * self.k + 1) * 2
        self.reg_predict = nn.SequentialCell([
            nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            nn.Conv2d(32, self.reg_out, kernel_size=1, stride=1, padding=0, pad_mode='pad', has_bias=True)
        ])

    # def load_model(self, model_path):
    #     print('Loading from {}'.format(model_path))
    #     state_dict = load_checkpoint(model_path)
    #     self.load_param_into_net(state_dict, strict_load=True)

    def construct(self, x):
        
        P3, P4, P5 = self.fpn(x)

        print(P3.shape)
        print(P4.shape)
        print(P5.shape)

        cls_predict = [self.cls_predict(P3), self.cls_predict(P4), self.cls_predict(P5)]
        reg_predict = [self.reg_predict(P3), self.reg_predict(P4), self.reg_predict(P5)]

        return cls_predict, reg_predict

    def construct_test(self, x):
        P3, P4, P5 = self.fpn(x)
        cls_predict = [self.cls_predict(P3), self.cls_predict(P4), self.cls_predict(P5)]
        reg_predict = [self.reg_predict(P3), self.reg_predict(P4), self.reg_predict(P5)]

        return cls_predict, reg_predict


if __name__ == '__main__':
    import numpy

    x = Tensor(numpy.random.randn(4, 3, 512, 512), mstype.float32)
    print(x.shape)
    network = TextNet(k=3)
    cls_predict, reg_predict = network(x)
    print(cls_predict[0].shape, reg_predict[0].shape)
    print(cls_predict[1].shape, reg_predict[1].shape)
    print(cls_predict[2].shape, reg_predict[2].shape)
