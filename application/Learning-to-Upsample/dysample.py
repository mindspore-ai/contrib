import mindspore
import mindspore.numpy as numpy
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore import Tensor, context, dtype as mstype
import numpy as np

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        mindspore.common.initializer.Normal(std, mean)
    if hasattr(module, 'bias') and module.bias is not None:
        mindspore.common.initializer.Constant(bias)(module.bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        mindspore.common.initializer.Constant(val)(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        mindspore.common.initializer.Constant(bias)(module.bias)

class DySample(nn.Cell):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, has_bias=False)
            constant_init(self.scope, val=0.)
        
        self.A = mindspore.Parameter(self._init_pos(), name='init_pos', requires_grad=False)

    def _init_pos(self):
        h = numpy.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        inputs = (h, h)
        grid = ops.Stack()(ops.Meshgrid()(inputs))
        grid = ops.transpose(grid, (1, 2, 0))
        grid = ops.tile(grid, (1, self.groups, 1, 1))
        grid = ops.reshape(grid, (1, -1, 1, 1))
        return grid

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = ops.reshape(offset, (B, 2, -1, H, W))
        coords_h = Tensor(list(range(H)), mstype.float32) + 0.5
        coords_w = Tensor(list(range(W)), mstype.float32) + 0.5
        inputs = (coords_h, coords_w)       
        coords = ops.Stack()(ops.Meshgrid(indexing='xy')(inputs))
        coords = ops.transpose(coords, (0, 2, 1))
        coords = ops.expand_dims(ops.expand_dims(coords, 1), 0)
        coords = coords.astype(x.dtype)
        normalizer = Tensor([W, H], mstype.float32).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = ops.pixel_shuffle(ops.reshape(coords, (B, -1, H, W)), self.scale)
        coords = ops.reshape(coords, (B, 2, -1, self.scale * H, self.scale * W))
        coords = ops.transpose(coords, (0, 2, 3, 4, 1))
        coords = ops.flatten(coords, start_dim = 0,end_dim = 1)
        return ops.grid_sample(ops.reshape(x, (B * self.groups, -1, H, W)), coords, mode='bilinear', 
                               align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)


    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.A
        else:
            offset = self.offset(x) * 0.25 + self.A
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = ops.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = ops.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.A
            offset = ops.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.A
        return self.sample(x, offset)

    def construct(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


if __name__ == '__main__':
    random_array = np.random.rand(2, 64, 4, 7).astype(np.float32)
    x = mindspore.Tensor(random_array, mindspore.dtype.float32)
    dys = DySample(64)
    print(dys(x).shape)