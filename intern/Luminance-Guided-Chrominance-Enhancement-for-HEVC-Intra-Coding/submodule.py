import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

def conv1x1(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=1, stride=stride,pad_mode = 'pad' ,padding=0, has_bias=bias)

def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=3, stride=stride,pad_mode = 'pad', padding=1, has_bias=bias)

def gate():
    return mindspore.Parameter(Tensor([1]))

class AU(nn.Cell):
    """Asymmetric Unit"""
    def __init__(self, n):
        super(AU, self).__init__()
        self.b1 = nn.SequentialCell(
            conv1x1(n,n),
            nn.LeakyReLU(alpha = 0.01))
        self.b2 = nn.SequentialCell(
            conv1x1(n,n),
            nn.LeakyReLU(alpha = 0.01),
            conv3x3(n,n), 
            nn.LeakyReLU(alpha = 0.01))
        self.b3 = nn.SequentialCell(
            conv1x1(n,n),
            nn.LeakyReLU(alpha = 0.01),
            nn.AvgPool2d(kernel_size=3,stride=1,pad_mode ='pad',padding=1),
            nn.LeakyReLU(alpha = 0.01))
        self.b4 = nn.SequentialCell(
            conv3x3(n,n),
            nn.LeakyReLU(alpha = 0.01))
    def construct(self, x):
        y = self.b1(x)+self.b2(x)+self.b3(x)+self.b4(x)
        return y
        
class AB(nn.Cell):
    def __init__(self, n):
        super(AB, self).__init__()
        self.au1 = AU(n)
        self.au2 = AU(n)
        self.tail = conv3x3(n,n)
    def construct(self, x):
        x1 = self.au1(x)
        x2 = self.au2(x1)
        x3 = self.au2(x2+x1)
        x4 = self.au2(x3+x1)
        y = self.tail(x4)
        return y

class FEB(nn.Cell):
    def __init__(self, in_channels, feature_channels):
        super(FEB, self).__init__()
        self.c1 = nn.SequentialCell(conv3x3(in_channels,feature_channels), nn.ReLU())
        self.c2 = nn.SequentialCell(conv3x3(feature_channels,feature_channels), nn.ReLU())
        self.c3 = nn.SequentialCell(conv3x3(feature_channels,feature_channels), nn.ReLU())
        self.c4 = nn.SequentialCell(conv3x3(feature_channels,feature_channels), nn.ReLU())
    def construct(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        return x1 + x2 + x3 + x4

