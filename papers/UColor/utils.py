"""
Some utils for ucolor
"""

import logging
import os

import numpy as np
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.dataset.vision.py_transforms import RgbToHsv


def tensor2image(tensor):
    """transfer the tensor into numpy"""
    img = tensor.asnumpy()
    img *= 255
    img = img.clip(0, 255)
    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))
    return img


def split(x):
    """split for input"""
    if len(x.shape) == 4:
        return x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9::, :, :]
    return x[0:3, :, :], x[3:6, :, :], x[6:9, :, :], x[9::, :, :]


def conv2d(input_dim, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2,
           stddev=0.02):
    """make the conv"""
    return nn.Conv2d(input_dim, output_dim,
                     kernel_size=(k_h, k_w), stride=(d_h, d_w),
                     pad_mode='same', has_bias=True,
                     weight_init=TruncatedNormal(sigma=stddev), bias_init='zeros')


class RgbToLab:
    """
    Domain tansfer util
    """
    def __init__(self):
        self.rgb_to_xyz_mat = np.array([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169, 0.950227],  # B
        ])
        self.fxfyfz_to_lab = np.array([
            #  l       a       b
            [0.0, 500.0, 0.0],  # fx
            [116.0, -500.0, 200.0],  # fy
            [0.0, 0.0, -200.0],  # fz
        ])

    def rgb2lab(self, x):
        """transfer rgb to lab"""
        x = x.transpose((1, 2, 0))
        pixels = np.reshape(x, [-1, 3])
        linear_mask = (pixels <= 0.04045).astype(np.float32)
        exponential_mask = (pixels > 0.04045).astype(np.float32)
        rgb_pixels = (pixels / 12.92 * linear_mask) + \
                    (((pixels + 0.055) / 1.055) ** 2.4) * \
                    exponential_mask
        xyz_pixels = np.matmul(rgb_pixels, self.rgb_to_xyz_mat)

        xyz_normalized_pixels = xyz_pixels * \
            np.array([1 / 0.950456, 1.0, 1 / 1.088754]).reshape((1, 3))
        epsilon = 6 / 29
        linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).astype(np.float32)
        exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).astype(np.float32)
        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + \
            (xyz_normalized_pixels ** (1 / 3)) * exponential_mask

        lab_pixels = np.matmul(fxfyfz_pixels, self.fxfyfz_to_lab) + \
            np.array([-16.0, 0.0, 0.0]).reshape((1, 3))

        y = np.reshape(lab_pixels, x.shape).transpose((2, 0, 1))

        # 0 - 1 uniform
        l = y[0:1, :, :]
        ab = y[1::, :, :]
        l = l / 100
        ab = (ab + 128) / 255
        y = np.concatenate([l, ab], axis=0)
        return y

    def __call__(self, rgb_img):
        return self.rgb2lab(rgb_img)


class DataTransform:
    """
    transfer the data
    """
    def __init__(self):
        self.rgb2hsv = RgbToHsv()
        self.rgb2lab = RgbToLab()

    def __call__(self, x):
        rgb, hsv, lab, depth = split(x)
        hsv = self.rgb2hsv(hsv)
        hsv = hsv.astype(np.float32)
        lab = self.rgb2lab(lab)
        lab = lab.astype(np.float32)

        return np.concatenate([rgb, hsv, lab, depth], axis=0)

class AverageUtil:
    """to save the avg"""
    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        """reset the util"""
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, value, *, count=1):
        """update the avg"""
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count


class Log:
    """log"""
    def __init__(self, filename):
        assert not os.path.exists(filename), "log file: {filename} exists!"
        logging.basicConfig(filename=filename, level=logging.INFO)

    def __call__(self, info):
        logging.info(info)
        print(info)
