import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor

class GlobalMaxPool2d(nn.Cell):

    def __init__(self, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(GlobalMaxPool2d, self).__init__()
        self.stride = stride or 1
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'global max pooling, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, input):
        kernel_size = input.size(-1)
        return nn.max_pool2d(input, kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

class GlobalAvgPool2d(nn.Cell):

    def __init__(self, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(GlobalAvgPool2d, self).__init__()
        self.stride = stride or 1
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'global avg pooling'

    def forward(self, input):
        # kernel_size = input.size(-1)
        return mindspore.ops.AdaptiveAvgPool2D(input, 1)


class Split(nn.Cell):

    def __init__(self, splits):
        super(Split, self).__init__()
        self.splits = splits

    def extra_repr(self):
        return 'split layer, splits={splits}'.format(**self.__dict__)

    def forward(self, input):
        splits = np.cumsum([0] + self.splits)
        xs = [input[:,splits[i]:splits[i+1],:,:].contiguous() for i in range(len(splits) - 1)]
        return xs
