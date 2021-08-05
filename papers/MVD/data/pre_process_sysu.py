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
"""pre_process_sysu"""

import os
import numpy as np
from PIL import Image


# todo_change your own path
data_path = '/home/datasets/prml/computervision/re-id/sysu-mm01/ori_data'

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# load id info
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
file_path_val = os.path.join(data_path, 'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# combine train and val split
id_train.extend(id_val)

files_rgb = []
files_ir = []
for ide in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, ide)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, ide)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image):
    """read_imgs"""
    train_data = []
    labels = []
    for ipath in train_image:
        # img
        img = Image.open(ipath)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)
        train_data.append(pix_array)

        # label
        pid_label = int(ipath[-13:-9])
        pid_label = pid2label[pid_label]
        labels.append(pid_label)

    return np.array(train_img), np.array(train_label)


# rgb imges
train_img, train_label = read_imgs(files_rgb)
np.save(data_path + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# ir imges
train_img, train_label = read_imgs(files_ir)
np.save(data_path + 'train_ir_resized_img.npy', train_img)
np.save(data_path + 'train_ir_resized_label.npy', train_label)
