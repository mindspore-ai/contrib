import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class JeffreysLoss(nn.Cell):
    def __init__(self, weight=None, reduction='mean', coeff1=0.0, coeff2=0.0):
        super(JeffreysLoss, self).__init__()
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: mindspore.Tensor, n_classes: int, coeff1=0.0, coeff2=0.0):
        assert 0 <= coeff1 < 1
        assert 0 <= coeff2 < 1

        one_hot = ops.OneHot()
        on_value = Tensor(1.0 - coeff1 - coeff2, mindspore.float32)
        off_value = Tensor(coeff1 / (n_classes - 1), mindspore.float32)
        targets = one_hot(targets, n_classes, on_value, off_value)
        return targets

    @staticmethod
    def _jeffreys_one_cold(targets: mindspore.Tensor, n_classes: int):
        one_hot = ops.OneHot()
        on_value = Tensor(0, mindspore.float32)
        off_value = Tensor(1, mindspore.float32)
        targets = one_hot(targets, n_classes, on_value, off_value)
        return targets

    @staticmethod
    def _jeffreys_one_hot(targets: mindspore.Tensor, n_classes: int):
        one_hot = ops.OneHot()
        on_value = Tensor(1, mindspore.float32)
        off_value = Tensor(0, mindspore.float32)
        targets = one_hot(targets, n_classes, on_value, off_value)
        return targets

    def construct(self, inputs: mindspore.Tensor, targets: mindspore.Tensor):
        targets1 = JeffreysLoss._smooth_one_hot(targets, inputs.shape[-1], self.coeff1, self.coeff2)
        sm = ops.softmax(inputs, -1)
        lsm = ops.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        # (cross-entropy)  & (Jeffreys part 1=label-smoothing)
        loss = -(targets1 * lsm).sum(-1)

        # Jeffreys part 2
        lsmsm = lsm * sm
        targets21 = JeffreysLoss._jeffreys_one_cold(targets, inputs.shape[-1], )
        loss1 = (targets21 * lsmsm).sum(-1)

        targets22 = JeffreysLoss._jeffreys_one_hot(targets, inputs.shape[-1], )
        loss2 = (targets22 * sm).sum(-1)

        loss3 = loss1 / (ops.ones_like(loss2) - loss2)

        loss3 *= self.coeff2
        loss = loss + loss3
        if self.reduction == 'sum':
            loss = ops.sum(loss)
        elif self.reduction == 'mean':
            loss = ops.mean(loss)
        return loss

if __name__ == '__main__':
    criterion = JeffreysLoss(coeff1=0.1, coeff2=0.025)
    l1 = mindspore.Tensor([[0, 0, 1], [1, 0, 0], [0, 0, 1]], mindspore.float32)
    l2 = mindspore.Tensor([2, 2, 2], mindspore.int32)
    r = criterion(l1, l2)
    print(r)
