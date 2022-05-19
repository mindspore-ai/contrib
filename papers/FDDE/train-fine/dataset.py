# Copyright 2022 Huawei Technologies Co., Ltd
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
"""dataset"""

import os
import cv2
import mindspore
from mindspore import ops
import numpy as np


########################### Data Augmentation ###########################
class Normalize():
    """Data Augmentation"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255, body / 255, detail / 255

##########randomcrop##################
class RandomCrop():
    """Data Augmentation"""
    def __call__(self, image, mask=None, body=None, detail=None):
        h, w, _ = image.shape
        randw = np.random.randint(w / 8)
        randh = np.random.randint(h / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, h + offseth - randh, offsetw, w + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], body[p0:p1, p2:p3], detail[p0:p1, p2:p3]

###################RandomFlip##################
class RandomFlip():
    """Data Augmentation"""
    def __call__(self, image, mask=None, body=None, detail=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), body[:, ::-1].copy(), detail[:, ::-1].copy()

        if mask is None:
            return image
        return image, mask, body, detail

##################Resize############################
class Resize():
    """Data Augmentation"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        body = cv2.resize(body, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        detail = cv2.resize(detail, dsize=(self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return image, mask, body, detail

##########################ToTensor#################################
class ToTensor():
    """Data Augmentation"""
    def __call__(self, image, mask=None, body=None, detail=None):
        image = mindspore.from_numpy(image)
        image = ops.Transpose()(image, (2, 0, 1))
        if mask is None:
            return image
        mask = mindspore.from_numpy(mask)
        body = mindspore.from_numpy(body)
        detail = mindspore.from_numpy(detail)
        return image, mask, body, detail


########################### Config File ###########################
class Config():
    """Data Augmentation"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        return None


########################### Dataset Class ###########################
class Data():
    """Dataset Class"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        if self.cfg.mode == "train":
            self.samples = [os.path.splitext(f)[0] for f in os.listdir(cfg.datapath + '/image') if f.endswith('.jpg')]
        else:
            self.samples = [os.path.splitext(f)[0] for f in os.listdir(cfg.datapath) if f.endswith('.jpg')]


    def __getitem__(self, idx):
        name = self.samples[idx]
        if self.cfg.mode == "train":
            image = cv2.imread(self.cfg.datapath + '/image/' + name + '.jpg', cv2.IMREAD_COLOR).astype(np.float32)
            mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
            return image, mask
        image = cv2.imread(self.cfg.datapath + '/' + name + '.jpg', cv2.IMREAD_COLOR).astype(np.float32)
        mask = cv2.imread(self.cfg.datapath + '/' + name + '.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return image, mask, idx, mask.shape


    def __len__(self):
        return len(self.samples)
