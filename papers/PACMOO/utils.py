"""
:author: hxq, shy
"""
import numpy as np
from mindspore import ops

DIV = ops.Div()


def l2_regularizer(input_x, axis=1):
    """
    :param input_x:
    :param axis:
    :return:
    """
    return DIV(input_x, ((input_x**2).sum(axis=axis, keepdims=True))**0.5)


def set_rng_seed(seed):
    """
    :param seed:
    :return:
    """
    np.random.seed(seed)
