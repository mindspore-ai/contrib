"""
     loss.py
"""

import mindspore.ops as ops
from mindspore.nn.loss.loss import _Loss
import mindspore.nn as nn
from losses.contras_loss import ContrastLoss

ops_print = ops.Print()
class Loss(_Loss):
    """
        Loss
    """

    def __init__(self):
        super(Loss, self).__init__()
        self.l1loss = nn.L1Loss()
        self.contras_loss = ContrastLoss()
        self.w_loss_l1 = 1
        self.w_loss_vgg7 = 0.1

    def construct(self, pred, pos, neg, gt):
        l1_loss = self.l1loss(pred, gt)
        contras_loss = self.contras_loss(pred, pos, neg)

        loss = self.w_loss_l1 * l1_loss + self.w_loss_vgg7 * contras_loss
        return self.get_loss(loss)

class CustomWithLossCell(nn.Cell):
    """
     CustomWithLossCell
    """
    def __init__(self, net, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, data, pos, neg, gt):
        output = self._backbone(data)
        return self._loss_fn(output, pos, neg, gt)

# net = DehazeNet()
# loss = Loss()
# loss_net = CustomWithLossCell(net, loss)
