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
"""data_manager"""

from __future__ import print_function, absolute_import
import os
import random
import numpy as np


def process_query_sysu(data_path, mode='all', relabel=False):
    """process_query_sysu"""
    ir_cameras = ['cam3', 'cam6']

    if mode == 'indoor':
        ir_cameras = ['cam3', 'cam6']

    print(relabel)
    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for ide in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, ide)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if i[0] != '.'])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []

    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False):
    """process_gallery_sysu"""
    random.seed(trial)
    rgb_cameras = ['cam1', 'cam2']

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']

    print(relabel)
    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for ide in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, ide)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if i[0] != '.'])
                files_rgb.append(random.choice(new_files))

    gall_img = []
    gall_id = []
    gall_cam = []

    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    """process_test_regdb"""
    input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'

    with open(input_data_path, "rt") as f:
        data_file_list = f.read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)
