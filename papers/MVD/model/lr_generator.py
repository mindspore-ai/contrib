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
"""lr_generator"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops as P


def _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    """
    Applies liner decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr)
    return lr_each_step


def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, poly, cosine or liner(default)

    Returns:
       np.array, learning rate array
    """
    print(lr_decay_mode)
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    lr_each_step = _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


class LearningRateSchedule(nn.Cell):
    """Basic class of learning rate schedule."""

    def construct(self, global_step):
        """
        Defines the computation to get the current learning rate.

        This method must be overridden by all subclasses.

        Note:
            The output must be a Tensor of scalar.

        Inputs:
            Tensor. The current step number.
        """
        raise NotImplementedError


class LRScheduler(LearningRateSchedule):
    r"""
    Gets learning rate warming up + decay.

    Args:
        learning_rate (float): The initial value of learning rate.
        warmup_steps (int): The warm up steps of learning rate.
        weight_decay (int): The weight decay steps of learning rate.

    Inputs:
        Tensor. The current step number.

    Outputs:
        Tensor. The learning rate value for the current step.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, learning_rate, weight_decay, warmup_steps=0):
        super(LRScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min = P.Minimum()
        self.cast = P.Cast()

    def construct(self, global_step):
        if global_step < self.warmup_steps:
            warmup_percent = self.cast(self.min(global_step, self.warmup_steps), ms.float32) / self.warmup_steps
            return self.learning_rate * warmup_percent
        lr = self.learning_rate
        for decay in self.weight_decay:
            if global_step <= decay:
                break
            lr = lr * 0.1
        lr = self.cast(lr, ms.float32)
        return lr
