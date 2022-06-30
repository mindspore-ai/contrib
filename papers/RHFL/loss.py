"""
Filename: loss.py
Author: zhangjiaming
Contact: 1692823208@qq.com
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

eps = 1e-7


class SCELoss(nn.Cell):
    """
    SCELoss
    """
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.softmax = ops.Softmax(axis=1)
        self.onehot = ops.OneHot()
        self.log = ops.Log()
        self.sum = ops.ReduceSum()

    def construct(self, pred, labels):
        """

        Args:
            pred:
            labels:

        Returns:

        """

        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = self.softmax(pred)
        pred = ops.clip_by_value(pred, clip_value_min=Tensor(1e-7), clip_value_max=Tensor(1.0))
        label_one_hot = self.onehot(labels, self.num_classes, Tensor(1.0),
                                    Tensor(0.0))
        label_one_hot = ops.clip_by_value(label_one_hot, clip_value_min=Tensor(1e-4), clip_value_max=Tensor(1.0))
        rce = (-1 * self.sum(pred * self.log(label_one_hot), 1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class KLDivLoss(nn.Cell):
    """
    KLDivLoss
    """

    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.kl = ops.KLDivLoss(reduction='sum')

    def construct(self, base, target):
        """

        Args:
            base:
            target:

        Returns:

        """
        loss = self.kl(base, target) / base.shape[0]
        return loss
