import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import _selected_ops


class LossNet(nn.Cell):
    """modified loss function"""

    def __init__(self, temp=0.1):
        super(LossNet, self).__init__()
        self.concat = P.Concat()
        self.exp = P.Exp()
        self.t = P.Transpose()
        self.diag_part = P.DiagPart()
        self.matmul = P.MatMul()
        self.sum = P.ReduceSum()
        self.sum_keep_dim = P.ReduceSum(keep_dims=True)
        self.log = P.Log()
        self.mean = P.ReduceMean()
        self.shape = P.Shape()
        self.eye = P.Eye()
        self.temp = temp

    def diag_part_new(self, input, batch_size):
        eye_matrix = self.eye(batch_size, batch_size, mindspore.float32)
        input = input * eye_matrix
        input = self.sum_keep_dim(input, 1)
        return input

    def construct(self, x, y, z_aux, label):
        batch_size = self.shape(x)[0]
        embed_size = self.shape(x)[1]

        perm = (1, 0)
        mat_x_x = self.exp(self.matmul(x, self.t(x, perm) / self.temp))
        mat_y_y = self.exp(self.matmul(y, self.t(y, perm) / self.temp))
        mat_x_y = self.exp(self.matmul(x, self.t(y, perm) / self.temp))

        mat_aux_x = self.exp(self.matmul(x, self.t(z_aux, perm) / self.temp))
        mat_aux_y = self.exp(self.matmul(y, self.t(z_aux, perm) / self.temp))
        mat_aux_z_x = self.exp(self.matmul(z_aux, self.t(x, perm) / self.temp))
        mat_aux_z_y = self.exp(self.matmul(z_aux, self.t(y, perm) / self.temp))

        loss_mutual = self.mean(-2 * self.log(self.diag_part_new(mat_x_y, batch_size) / (
                    self.sum_keep_dim(mat_x_y, 1) - self.diag_part_new(mat_x_y, batch_size) +
                    self.sum_keep_dim(mat_x_x,1) - self.diag_part_new(mat_x_x, batch_size) +
                    self.sum_keep_dim(mat_y_y, 1) - self.diag_part_new(mat_y_y, batch_size))))

        loss_aux_x = self.mean(-self.log((self.diag_part_new(mat_aux_x, batch_size) / (
                self.sum_keep_dim(mat_aux_x, 1) - self.diag_part_new(mat_aux_x, batch_size)))))

        loss_aux_y = self.mean(-self.log((self.diag_part_new(mat_aux_y, batch_size) / (
                self.sum_keep_dim(mat_aux_y, 1) - self.diag_part_new(mat_aux_y, batch_size)))))

        loss_aux_z_x = self.mean(- self.log((self.diag_part_new(mat_aux_z_x, batch_size) / (
                self.sum_keep_dim(mat_aux_z_x, 1) - self.diag_part_new(mat_aux_z_x, batch_size)))))

        loss_aux_z_y = self.mean(-self.log((self.diag_part_new(mat_aux_z_y, batch_size) / (
                self.sum_keep_dim(mat_aux_z_y, 1) - self.diag_part_new(mat_aux_z_y, batch_size)))))

        loss = loss_mutual + loss_aux_x + loss_aux_y + loss_aux_z_x + loss_aux_z_y

        return loss


class _Loss(nn.Cell):
    """
    Base class for other losses.
    """

    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = _selected_ops.ReduceMean()
        self.reduce_sum = P.ReduceSum()

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x):
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        return x

    def construct(self, base, target):
        raise NotImplementedError


class BCELoss(_Loss):
    r"""
    BCELoss creates a criterion to measure the Binary Cross Entropy between the true labels and predicted labels.

    Note:
        Set the predicted labels as :math:`x`, true labels as :math:`y`, the output loss as :math:`\ell(x, y)`.
        Let,

        .. math::
            L = \{l_1,\dots,l_N\}^\top, \quad
            l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

        Then,

        .. math::
            \ell(x, y) = \begin{cases}
            L, & \text{if reduction} = \text{`none';}\\
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
            \end{cases}

        Note that the predicted labels should always be the output of sigmoid and the true labels should be numbers
        between 0 and 1.

    Args:
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            And it must have same shape and data type as `inputs`. Default: None
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.

    Inputs:
        - **inputs** (Tensor) - The input Tensor. The data type must be float16 or float32.
        - **labels** (Tensor) - The label Tensor which has same shape and data type as `inputs`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `inputs`.
        Otherwise, the output is a scalar.

    Examples:
        >>> weight = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 3.3, 2.2]]), mindspore.float32)
        >>> loss = nn.BCELoss(weight=weight, reduction='mean')
        >>> inputs = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]), mindspore.float32)
        >>> loss(inputs, labels)
        1.8952923
    """

    def __init__(self, weight=None, reduction='none'):
        super(BCELoss, self).__init__()
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction=reduction)
        self.weight_one = weight is None
        if not self.weight_one:
            self.weight = weight
        else:
            self.ones = P.OnesLike()

    def construct(self, inputs, labels):
        if self.weight_one:
            weight = self.ones(inputs)
        else:
            weight = self.weight
        loss = self.binary_cross_entropy(inputs, labels, weight)
        return loss
