import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor


class Expand(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        return ops.unsqueeze(x, dim=0)


class Squeeze(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        res = ops.squeeze(x, axis=1)
        return res


class SE_block(nn.Cell):
    """Squeeze Excite block"""
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_2d = ops.adaptive_avg_pool2d
        self.dense_block = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // ratio, has_bias=False),
            nn.ReLU(),
            nn.Dense(in_channels // ratio, in_channels, has_bias=False),
            nn.Sigmoid(),
        )

    def construct(self, x):
        filters = x.shape[1]
        reshape_size = (x.shape[0], 1, 1, filters)
        se = self.avg_2d(x, (1, 1))
        se = ops.Reshape()(se, reshape_size)
        se = self.dense_block(se)
        se = ops.Transpose()(se, (0, 3, 1, 2))
        return x * se


class BN_block2d(nn.Cell):
    """2-d batch-norm block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):
        return self.bn_block(x)


class BN_block3d(nn.Cell):
    """3-d batch-norm block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block = nn.SequentialCell([
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ])

    def construct(self, x):
        return self.bn_block(x)


class D_SE_Add(nn.Cell):
    """D_SE_Add block"""
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.conv3d_ = nn.Conv3d(in_channels, 1, kernel_size=1, padding=0, has_bias=True)
        self.Squeeze = Squeeze()
        self.conv2d_ = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        self.ReLU = nn.ReLU()
        self.SE_block = SE_block(in_channels)
        self.squeeze_block_3d = nn.SequentialCell([
            nn.Conv3d(in_channels, 1, kernel_size=1, padding=0, has_bias=True),
            Squeeze(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.ReLU(),
            SE_block(in_channels)
        ])

    def construct(self, in_3d, in_2d):
        in_2d = self.SE_block(in_2d)
        in_3d = self.squeeze_block_3d(in_3d)
        return in_3d + in_2d


def up_block(in_channels, out_channels):
    return nn.SequentialCell([
        nn.Upsample(scale_factor=2.0, mode='nearest', recompute_scale_factor=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
        nn.ReLU()
    ])
