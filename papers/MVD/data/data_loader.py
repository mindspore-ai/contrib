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
"""data_loader"""

import os
import numpy as np
from PIL import Image


class SYSUDatasetGenerator:
    """
    SYSUDatasetGenerator
    """

    def __init__(self, data_dir="./data/sysu/", transform_rgb=None, transform_ir=None, color_index=None,
                 thermal_index=None, if_debug=False):

        # Load training images (path) and labels
        if if_debug:
            self.train_color_image = np.load(os.path.join(data_dir, 'demo_train_rgb_resized_img.npy'))
            self.train_color_label = np.load(os.path.join(data_dir, 'demo_train_rgb_resized_label.npy'))

            self.train_thermal_image = np.load(os.path.join(data_dir, 'demo_train_ir_resized_img.npy'))
            self.train_thermal_label = np.load(os.path.join(data_dir, 'demo_train_ir_resized_label.npy'))
        else:
            self.train_color_image = np.load(os.path.join(data_dir, 'train_rgb_resized_img.npy'))
            self.train_color_label = np.load(os.path.join(data_dir, 'train_rgb_resized_label.npy'))

            self.train_thermal_image = np.load(os.path.join(data_dir, 'train_ir_resized_img.npy'))
            self.train_thermal_label = np.load(os.path.join(data_dir, 'train_ir_resized_label.npy'))

        print("Color Image Size:{}".format(len(self.train_color_image)))
        print("Color Label Size:{}".format(len(self.train_color_label)))

        self.transform_rgb = transform_rgb
        self.transform_ir = transform_ir
        self.cindex = color_index
        self.tindex = thermal_index

    def __next__(self):
        pass

    def __getitem__(self, index):
        # TODO: 这里要配合samplers输出的更改而更改
        # print(index)
        # print("self.cIndex is ",self.cIndex[index] )
        # print("self.tIndex is ",self.tIndex[index] )
        img1, target1 = self.train_color_image[self.cindex[index]], self.train_color_label[self.cindex[index]]
        img2, target2 = self.train_thermal_image[self.tindex[index]], self.train_thermal_label[self.tindex[index]]
        # print("img1 is:", img1)
        # print("target1 is:", target1)
        # print("img2 is:", img2)
        # print("target2 is:", target2)

        # img1, img2 = self.transform(img1)[0], self.transform(img2)[0]
        # target1, target2 = np.array(target1, dtype=np.float32), np.array(target2, dtype=np.float32)

        # img1, img2 = self.transform(img1)[0], self.transform(img2)[0]
        # img1, img2 = ms.Tensor(img1, dtype=ms.float32), ms.Tensor(img2, dtype=ms.float32)
        # target1, target2 = ms.Tensor(target1, dtype=ms.float32), ms.Tensor(target2, dtype=ms.float32)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.cindex)


class TestData:
    """TestData"""
    def __init__(self, test_img_file, test_label, img_size=(144, 288), transform=None):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label

        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        return img1, target1

    def __len__(self):
        return len(self.test_image)
