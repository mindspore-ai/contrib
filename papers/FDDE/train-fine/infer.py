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
"""infer"""



import dataset
from net import FDDE

import cv2
import numpy as np

import mindspore

from mindspore import load_checkpoint, load_param_into_net

from mindspore import context
import mindspore.nn as nn
from mindspore.nn import LossBase

import mindspore.dataset as ds

from mindspore.dataset.transforms.py_transforms import Compose

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class IoULoss(LossBase):
    """iou"""

    def construct(self, pred, mask):
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        inter = (pred * mask).sum(axis=(2, 3))
        union = (pred + mask).sum(axis=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        return iou.mean()


# dataset
########################### Data Augmentation ###########################
def Normalize(image, mask):
    mean = np.array([[[124.55, 118.90, 102.94]]])
    std = np.array([[[56.77, 55.97, 57.50]]])
    image = (image - mean) / std
    if mask is None:
        return image
    return image, mask / 255


def RandomCrop(image, mask):
    h, w, _ = image.shape
    randw = np.random.randint(w / 8)
    randh = np.random.randint(h / 8)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth, h + offseth - randh, offsetw, w + offsetw - randw
    if mask is None:
        return image[p0:p1, p2:p3, :]
    return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


def RandomFlip(image, mask):
    if np.random.randint(2) == 0:
        if mask is None:
            return image[:, ::-1, :].copy()
        return image[:, ::-1, :].copy(), mask[:, ::-1].copy()

    if mask is None:
        return image
    return image, mask


def Transpose(image, mask):
    h = 352
    w = 352
    image = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))
    mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
    mask = np.expand_dims(mask, axis=0)
    image = image.copy()
    mask = mask.copy()
    return image, mask


def Test(dataset_in, network):
    """test"""
    cfg = dataset_in.Config(datapath='/home/user/newdisk/wangchaowei/RGB/dataset/Pascal-S',
                            savepath='./outimg/Pascal-S',
                            mode='test', batch=1, lr=0.05, momen=0.9, decay=5e-4, epoch=40)
    data = dataset_in.Data(cfg)
    datasets = ds.GeneratorDataset(data, column_names=["image", "mask", "idx", "o_shape"])

    transforms_list = [Normalize, Transpose]
    compose_trans = Compose(transforms_list)
    datasets = datasets.map(operations=compose_trans, input_columns=["image", "mask"],
                            output_columns=["image", "mask"],
                            num_parallel_workers=4)

    datasets = datasets.batch(cfg.batch)

    ## load network
    net = network(cfg)
    param_dict = load_checkpoint("./out/model-39.ckpt")
    load_param_into_net(net, param_dict)
    net.set_train(False)

    name_samples = data.samples
    step_i = 0

    for in_c in datasets.create_dict_iterator():
        image, mask = in_c["image"], in_c["mask"]
        image, mask = image.astype(mindspore.dtype.float32), mask.astype(mindspore.dtype.float32)
        name = name_samples[in_c["idx"][0]]

        x_shape = np.squeeze(in_c["o_shape"].asnumpy())
        h, w = int(x_shape[0]), int(x_shape[1])
        _, out = net(image, (h, w))

        pred = nn.Sigmoid()(out).asnumpy() * 255
        pred = np.squeeze(pred).astype(np.uint8)
        print('{0}/{1} name: {2}  shape:{3}'.format(step_i, len(name_samples), name, pred.shape))
        cv2.imwrite(cfg.savepath + "/" + name + ".png", pred)
        step_i += 1


if __name__ == '__main__':
    Test(dataset, FDDE)
