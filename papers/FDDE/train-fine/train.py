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
"""train"""


import dataset
from net import FDDE

import cv2
import mindspore
from mindspore import save_checkpoint
from mindspore import ParameterTuple
from mindspore import Tensor
from mindspore import context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import LossBase
import mindspore.dataset as ds
from mindspore.dataset.transforms.py_transforms import Compose
import numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


########################iou#############################
class IoULoss(LossBase):
    """iou"""

    def construct(self, pred, mask):
        sigmoid = nn.Sigmoid()
        pred = sigmoid(pred)
        inter = (pred * mask).sum(axis=(2, 3))
        union = (pred + mask).sum(axis=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        return iou.mean()


# binary_cross_entropy_with_logits 可以使用 BCEWithLogitsLoss代替
# CPU版本使用 BCELoss 代替

###################################WithLossCell###############################
class WithLossCell(nn.Cell):
    """   WithLossCell      """

    def __init__(self, net, auto_prefix=False):
        super(WithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        # self.kldivloss = ops.KLDivLoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.bce_w2 = ops.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()

    def construct(self, image, mask, edg):
        out_l1, out = self.net(image)
        e_loss = self.bce_with_logits_loss(out_l1, edg)
        # e_loss = self.BCEW2(out_l1,edg,weight=)
        o_loss = self.bce_with_logits_loss(out, mask)
        i_loss = self.iou_loss(out, mask)

        loss = e_loss + o_loss + i_loss
        return loss


################################TrainOneStepCell############################
class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""

    def __init__(self, net, optim, sens=1.0, auto_prefix=False):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.netloss = net

        self.netloss.set_grad()

        self.weights = ParameterTuple(net.trainable_params())

        self.optimizer = optim
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, image, mask, edg):
        weights = self.weights
        loss = self.netloss(image, mask, edg)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.netloss, weights)(image, mask, edg, sens)
        return ops.depend(loss, self.optimizer(grads))


# dataset
########################### Data Augmentation ###########################
def Normalize(image, mask):
    mean = np.array([[[124.55, 118.90, 102.94]]])
    std = np.array([[[56.77, 55.97, 57.50]]])
    image = (image - mean) / std
    if mask is None:
        return image
    return image, mask / 255


########################### Data Augmentation ###########################
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


########################### Data Augmentation ###########################
def RandomFlip(image, mask):
    if np.random.randint(2) == 0:
        if mask is None:
            return image[:, ::-1, :].copy()
        return image[:, ::-1, :].copy(), mask[:, ::-1].copy()

    if mask is None:
        return image
    return image, mask


########################### Data Augmentation ###########################
def Transpose(image, mask):
    h = 352
    w = 352
    image = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))
    mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
    mask = np.expand_dims(mask, axis=0)
    image = image.copy()
    mask = mask.copy()
    return image, mask


########################### train ###########################
def Train(dataset_in, network_in):
    """train"""
    cfg = dataset_in.Config(datapath='DUTS-TR', savepath='./out', mode='train', batch=32, lr=0.05,
                            momen=0.9,
                            decay=5e-4,
                            epoch=40)
    data = dataset_in.Data(cfg)
    datasets = ds.GeneratorDataset(data, column_names=["image", "mask"])

    transforms_list = [Normalize, RandomCrop, RandomFlip, Transpose]
    compose_trans = Compose(transforms_list)
    datasets = datasets.map(operations=compose_trans, input_columns=["image", "mask"],
                            output_columns=["image", "mask"],
                            num_parallel_workers=4)

    datasets = datasets.shuffle(buffer_size=cfg.batch * 10)
    datasets = datasets.batch(cfg.batch)
    datasets = datasets.repeat(1)

    ## network
    net = network_in(cfg)

    # 设置学习率为自然指数衰减
    learning_rate = cfg.lr
    decay_rate = 0.9
    step_per_epoch = datasets.get_dataset_size()
    total_step = step_per_epoch * cfg.epoch
    decay_epoch = 4
    natural_exp_decay_lr = nn.exponential_decay_lr(learning_rate, decay_rate, total_step,
                                                   step_per_epoch, decay_epoch)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=natural_exp_decay_lr,
                       weight_decay=cfg.decay,
                       momentum=cfg.momen, nesterov=True)

    network = WithLossCell(net)
    network = TrainOneStepCell(network, optimizer)
    network.set_train()



    for epoch in range(cfg.epoch):

        step = 0
        for in_c in datasets.create_dict_iterator():
            image, mask = in_c["image"], in_c["mask"]
            image, mask = image.astype(mindspore.dtype.float32), mask.astype(
                mindspore.dtype.float32)
            weight = Tensor(np.ones([1, 1, 3, 3]), mindspore.float32)
            conv2d = ops.Conv2D(out_channel=1, kernel_size=3, pad=1, pad_mode="pad")
            edg = conv2d(mask, weight)

            edg = edg.asnumpy()
            edg = np.where(edg > 5, np.zeros(edg.shape), edg)
            edg = np.where(edg != 0, np.ones(edg.shape), edg)
            edg = Tensor(edg, mindspore.float32)

            # 迭代
            output_loss = network(image, mask, edg)

            step += 1
            print('step : {0}, epoch : {1}/{2} , loss : {3}'.format(step, epoch + 1, cfg.epoch,
                                                                    output_loss))

        # 保存模型
        if epoch > cfg.epoch * 1 / 2:
            save_checkpoint(net, cfg.savepath + '/model-' + str(epoch) + ".ckpt")


if __name__ == '__main__':
    Train(dataset, FDDE)
