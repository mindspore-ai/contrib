#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""tools"""

import numpy as np
from PIL import Image
from mindspore import Tensor
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter


def save_image(img, img_path):
    """Save a numpy image to the disk
    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """
    if isinstance(img, Tensor):
        img = decode_image(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))
    img_pil = Image.fromarray(img)
    img_pil.save(img_path)


def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    img = img.asnumpy()[0]
    image_numpy = np.transpose(img, (1, 2, 0)) * 255.
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    return image_numpy


def create_lol_dataset(dataset, batch_size=32, repeat_size=1, num_workers=1):
    """
    create dataset for train or test
    """
    rescale = 1.0 / 255.0
    shift = 0.0
    # define map operations
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    transform_list = [rescale_op, hwc2chw_op]
    # apply map operations on images
    dataset = dataset.map(input_columns="lol", operations=transform_list, num_parallel_workers=num_workers)
    # apply DatasetOps
    buffer_size = 1000
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_size)
    return dataset


def create_iqa_testset(dataset, batch_size=1, repeat_size=1, num_workers=1):
    """
    create dataset for train or test
    """
    resize_height, resize_width = 256, 256
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    transform_list = [resize_op, rescale_op, hwc2chw_op]
    dataset = dataset.map(input_columns="x", operations=transform_list, num_parallel_workers=num_workers)
    dataset = dataset.map(input_columns="y", operations=transform_list, num_parallel_workers=num_workers)
    # apply DatasetOps
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat(repeat_size)
    return dataset
