import numpy as np

from typing import Union
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
class SeesawLossWithLogits(nn.Cell):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.

    Args:
    class_counts: The list which has number of samples for each class.
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: Union[list, np.array], p: float = 0.8):
        super().__init__()

        class_counts = ms.Tensor(class_counts,dtype=ms.float32)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = ops.ones((len(class_counts), len(class_counts)),ms.float32)
        self.s = ms.numpy.where(conditions, trues, falses)

        self.eps = 1.0e-6

    def construct(self, logits, targets):

        max_element, _ = ops.max(logits,axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = ops.exp(logits)
        denominator = (
                              (1 - targets)[:, None, :]
                              * self.s[None, :, :]
                              * ops.exp(logits)[:, None, :]).sum(axis=-1) \
                      + ops.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * ops.log(sigma + self.eps)).sum(-1)
        return loss.mean()


class DistibutionAgnosticSeesawLossWithLogits(nn.Cell):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.

    Args:
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 for default following the paper.
    """

    def __init__(self, p: float = 0.8):
        super().__init__()
        self.eps = 1.0e-6
        self.p = p
        self.s = None
        self.class_counts = None

    def construct(self, logits, targets):
        if self.class_counts is None:
            self.class_counts = targets.sum(axis=0) + 1  # to prevent devided by zero.
        else:
            self.class_counts += targets.sum(axis=0)

        conditions = self.class_counts[:, None] > self.class_counts[None, :]
        trues = (self.class_counts[None, :] / self.class_counts[:, None]) ** self.p
        falses = ops.ones((len(self.class_counts), len(self.class_counts)),ms.float32)
        self.s = ms.numpy.where(conditions, trues, falses)

        max_element, _ = ops.max(logits,axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = ops.exp(logits)
        denominator = (
                              (1 - targets)[:, None, :]
                              * self.s[None, :, :]
                              * ops.exp(logits)[:, None, :]).sum(axis=-1) \
                      + ops.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * ops.log(sigma + self.eps)).sum(-1)
        return loss.mean()


import unittest
from mindspore import Tensor
import numpy as np

class TestSeesawLossWithLogits(unittest.TestCase):
    def test_seesaw_loss_with_logits(self):
        # 初始化SeesawLossWithLogits对象
        class_counts = [10, 20, 30]
        loss_fn = SeesawLossWithLogits(class_counts)

        # 创建一些模拟数据
        logits = ms.Tensor(np.array([[1.0, 2.0, 3.0]]), dtype=ms.float32)
        targets = ms.Tensor(np.array([[0, 1, 0]]), dtype=ms.float32)

        # 计算损失
        loss = loss_fn(logits, targets)

        # 验证损失值是否正确
        # 由于是单元测试，实际值可能因真实数据和模型参数而异
        # 但我们可以检查loss是否被正确计算（例如，不为nan或inf）
        # Verify the loss is correct
        self.assertTrue(loss.item() > 0, "Loss should be a positive value")

class TestDistibutionAgnosticSeesawLossWithLogits(unittest.TestCase):
    def test_distibution_agnostic_seesaw_loss_with_logits(self):
        # 初始化DistibutionAgnosticSeesawLossWithLogits对象
        loss_fn = DistibutionAgnosticSeesawLossWithLogits()

        # 创建一些模拟数据
        logits = ms.Tensor(np.array([[1.0, 2.0, 3.0]]), dtype=ms.float32)
        targets = ms.Tensor(np.array([[0, 1, 0]]), dtype=ms.float32)

        # 计算损失
        loss = loss_fn(logits, targets)

        # 验证损失值是否正确
        # 由于是单元测试，实际值可能因真实数据和模型参数而异
        # 但我们可以检查loss是否被正确计算（例如，不为nan或inf）
        # Verify the loss is correct
        self.assertTrue(loss.item() > 0, "Loss should be a positive value")

if __name__ == '__main__':
    unittest.main()
