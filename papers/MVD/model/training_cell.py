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
"""training_cell"""

import os
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import ParameterTuple


def show_memory_info(hint=""):
    """show_memory_info"""
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")


class CriterionWithNet(nn.Cell):
    """CriterionWithNet"""
    def __init__(self, backbone, ce_loss, tri_loss, kl_div, t=1, lossfunc='id'):
        super(CriterionWithNet, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss
        self._kl_div = kl_div
        self.t = t
        self.softmax = nn.Softmax()
        self.lossfunc = lossfunc
        self.acc = 0

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def _get_acc(self, logits, label):
        predict, _ = self.max(logits)
        correct = self.eq(predict, label)
        return np.where(correct)[0].shape[0] / label.shape[0]

    def construct(self, img1, img2, label1, label2, modal=0, cpa=False):
        """build"""
        v_observation, v_representation, v_ms_observation, v_ms_representation, i_observation, i_representation, \
            i_ms_observation, i_ms_representation = self._backbone(img1, x2=img2, mode=0)
        print(modal, cpa)
        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        loss_id = 0.5 * (self._ce_loss(v_observation[1], label_) + self._ce_loss(v_representation[1], label_)) \
            + 0.5 * (self._ce_loss(i_observation[1], label_) + self._ce_loss(i_representation[1], label_)) \
            + 0.25 * (self._ce_loss(v_ms_observation[1], label_) + self._ce_loss(v_ms_representation[1], label_)) \
            + 0.25 * (self._ce_loss(i_ms_observation[1], label_) + self._ce_loss(i_ms_representation[1], label_))

        # fix loss_tri bug
        loss_tri = 0.5 * (self._tri_loss(v_observation[0], label) + self._tri_loss(v_representation[0], label)) \
            + 0.5 * (self._tri_loss(i_observation[0], label) + self._tri_loss(i_representation[0], label)) \
            + 0.25 * (self._tri_loss(v_ms_observation[0], label) +
                      self._tri_loss(v_ms_representation[0], label)) \
            + 0.25 * (self._tri_loss(i_ms_observation[0], label) + self._tri_loss(i_ms_representation[0], label))

        # vml, convert SinkhornDistance

        loss_total = 0
        for k in self.lossfunc.split("+"):

            if k == 'tri':
                loss_total += loss_tri
            if k == 'id':
                loss_total += loss_id

        self.acc = self._get_acc(v_observation[1], label_) + self._get_acc(v_representation[1], label_) \
            + self._get_acc(i_observation[1], label_) + self._get_acc(i_representation[1], label_) \
            + self._get_acc(v_ms_observation[1], label_) + self._get_acc(v_ms_representation[1], label_) \
            + self._get_acc(i_ms_observation[1], label_) + self._get_acc(i_ms_representation[1], label_)

        self.acc = self.acc / 8.0

        return loss_total

    @property
    def backbone_network(self):
        return self._backbone


class OptimizerWithNetAndCriterion(nn.Cell):
    """OptimizerWithNetAndCriterion"""
    def __init__(self, network, optimizer, sens=1.0):
        super(OptimizerWithNetAndCriterion, self).__init__(auto_prefix=True)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        # print(np.sum(self.network.trainable_params()[0].asnumpy()))
        # for i in range(len(self.network.trainable_params())):
        #     print(np.sum(self.network.trainable_params()[i].asnumpy()))
        return P.depend(loss, self.optimizer(grads))
