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
"""kaiming_normal"""


import math
import numpy as np

from mindspore.common.seed import get_seed, _get_graph_seed
from mindspore import Tensor
from mindspore import dtype as mstype

class Initializer:
    """
    The base class of the initializer.
    Initialization of tensor basic attributes and model weight values.

    Args:
        kwargs (dict): Keyword arguments for Initializer.

    Returns:
        Array, an array after being initialized.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._seed = None

    @property
    def seed(self):
        if self._seed is None:
            seed, seed2 = _get_graph_seed(get_seed(), "init")
        else:
            seed, seed2 = self._seed + 1, 0
        return seed, seed2

    @seed.setter
    def seed(self, value):
        self._seed = value

    def __call__(self, arr):
        return self._initialize(arr)

def _assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    num = Tensor(num, mstype.float32)
    arr.set_data(num)


def _calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(shape, mode):
    """
    Calculate fan.

    Args:
        shape (tuple): input shape.
        mode (str): only support fan_in and fan_out.

    Returns:
        fan_in or fan_out.
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    """
    Calculate gain.

    Args:
        nonlinearity (str): nonlinearity function.
        param (str): used to calculate negative_slope.

    Returns:
        number.
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res

def _calculate_in_and_out(arr):
    """
    Calculate n_in and n_out.

    Args:
        arr (Array): Input array.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dim = len(arr.shape)
    if dim < 2:
        raise ValueError("If initialize data with xavier uniform, the dimension of data must be greater than 1.")

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out

class HeNormal(Initializer):
    r"""
    Initialize the array with He kaiming Normal algorithm, and from a normal distribution collect samples within
    N(0, sigma).

    Args:
        negative_slope (int, float, bool): The negativa slope of the rectifier used after this layer
            (only used when `nonlinearity` is 'leaky_relu'). Default: 0.
        mode (str): Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
            variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes
            in the backwards pass. Default: fan_in.
        nonlinearity (str): The non-linear function, recommended to use only with 'relu' or 'leaky_relu'.
            Default: leaky_relu.

    Returns:
        Array, assigned array.
    """
    def __init__(self, negative_slope=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(HeNormal, self).__init__(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinearity)
        self.negative_slope = negative_slope
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        fan = _calculate_correct_fan(arr.shape, self.mode)
        gain = _calculate_gain(self.nonlinearity, self.negative_slope)
        std = gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)
