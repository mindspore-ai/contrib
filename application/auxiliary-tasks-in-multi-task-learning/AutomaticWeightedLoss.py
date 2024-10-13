import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
import numpy as np

class AutomaticWeightedLoss(nn.Cell):
    """Automatically weighted multi-task loss for MindSpore.

    Params:
        num: int, the number of losses.
        x: multi-task losses.
    Examples:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # Initialize the parameters with ones and set requires_grad to True
        params = Tensor(np.ones(num), mindspore.float32)
        self.params = Parameter(params, requires_grad=True)

    def construct(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + ops.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    for param in awl.get_parameters():
        print(param)
