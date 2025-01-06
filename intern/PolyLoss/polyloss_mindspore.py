import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.nn import LossBase
from mindspore.common import dtype as mstype


def to_one_hot(labels, num_classes, dtype=mstype.float32, dim=1):
    if len(labels.shape) < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = ops.reshape(labels, tuple(shape))

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError(
            "labels should have a channel with length equal to one."
        )

    sh[dim] = num_classes

    labels = ops.one_hot(
        labels.squeeze(dim).astype(mstype.int32),
        num_classes,
        Tensor(1.0, dtype),
        Tensor(0.0, dtype)
    )
    return labels


class PolyLoss(LossBase):
    def __init__(self, softmax=True, ce_weight=None, reduction='mean', epsilon=1.0):
        super(PolyLoss, self).__init__(reduction)
        self.softmax = softmax
        self.epsilon = epsilon
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='none'
        )
        self.ce_weight = ce_weight

    def construct(self, input, target):
        if len(input.shape) - len(target.shape) == 1:
            target = ops.expand_dims(target, 1).astype(mstype.int32)
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]

        if n_pred_ch != n_target_ch:
            self.ce_loss = self.cross_entropy(
                input, target.squeeze(1).astype(mstype.int32)
            )
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            self.ce_loss = self.cross_entropy(
                input, ops.argmax(target, axis=1)
            )

        if self.softmax and n_pred_ch != 1:
            input = ops.softmax(input, axis=1)

        pt = (input * target).sum(axis=1)
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        return self.get_loss(poly_loss)


class PolyBCELoss(LossBase):
    def __init__(self, reduction='mean', epsilon=1.0):
        super(PolyBCELoss, self).__init__(reduction)
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def construct(self, input, target):
        self.bce_loss = self.bce(input, target)
        pt = ops.sigmoid(input)
        pt = ops.where(target == 1, pt, 1 - pt)
        poly_loss = self.bce_loss + self.epsilon * (1 - pt)

        return self.get_loss(poly_loss)
