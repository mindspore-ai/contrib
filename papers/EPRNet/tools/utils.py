# coding=utf-8

import os
import numpy as np
from PIL import Image

__all__ = ['mask_color_to_gray', 'validate_ckpt']


def mask_color_to_gray(mask_gray_dir, mask_color_dir):
    if not os.path.exists(mask_gray_dir):
        os.makedirs(mask_gray_dir)

    for ann in os.listdir(mask_color_dir):
        ann_img = Image.open(os.path.join(mask_color_dir, ann))
        ann_img = Image.fromarray(np.array(ann_img))
        ann_img.save(os.path.join(mask_gray_dir, ann))


def validate_ckpt(*ckpt_paths):
    ckpt_path = os.path.join(*ckpt_paths)
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f'invalid checkpoint file: {ckpt_path}')
    return ckpt_path
