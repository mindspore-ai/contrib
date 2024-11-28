import math
from mindspore import nn, Parameter
from mindspore.common import Tensor
import mindspore as ms
import mindspore.ops as ops


class TSGD(nn.Optimizer):
    r"""Implements: Scaling transition from SGDM to plain SGD.
    'https://arxiv.org/abs/2106.06753'
    
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        iters(int): iterations
            iters = math.ceil(trainSampleSize / batchSize) * epochs
        lr (float): learning rate 
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        coeff(float, optional): scaling coefficient
    """

    def __init__(self, params, iters, momentum=0.9, dampening=0, weight_decay=0, nesterov=False, up_lr=5e-1, low_lr=5e-3, coeff1=1e-2, coeff2=5e-4):
        if not 1 <= iters:
            raise ValueError("Invalid iters: {}".format(iters))             
        if momentum <= 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= coeff1 <= 1.0:
            raise ValueError("Invalid coeff1: {}".format(coeff1))            
        if not 0.0 <= coeff2 <= 1.0:
            raise ValueError("Invalid coeff2: {}".format(coeff2))                      
        if not 0.0 <= up_lr:
            raise ValueError("Invalid up learning rate: {}".format(up_lr))
        if not 0.0 <= low_lr:
            raise ValueError("Invalid low learning rate: {}".format(low_lr))            
        if not low_lr <= up_lr:
            raise ValueError("required up_lr >= low_lr, but (up_lr = {}, low_lr = {})".format(up_lr, low_lr))                  

        super(TSGD, self).__init__(up_lr, params)
        self.iters = iters
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.up_lr = up_lr
        self.low_lr = low_lr
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.steps = self.parameters.clone(prefix="steps", init="zeros")

    def construct(self, grads):
        params = self.parameters
        for i, (param, grad) in enumerate(zip(params, grads)):
            if grad is None:
                continue
            if self.weight_decay != 0:
                grad += param * self.weight_decay
            step = self.steps[i]
            buf = self.moments[i]
            ops.assign(self.steps[i], self.steps[i] + 1)
            ops.assign(self.moments[i], self.moments[i] * self.momentum + grad * (1 - self.dampening))

            if self.nesterov:
                grad = grad + buf * self.momentum

            rho1 = 10 ** (math.log(self.coeff1, 10) / self.iters)
            rho2 = 10 ** (math.log(self.coeff2, 10) / self.iters)

            grad = (buf - grad) * (rho1 ** step) + grad
            lr = ((self.up_lr - self.low_lr) * rho2 ** step + self.low_lr) * (1 - rho2 ** step)

            param = ops.assign_sub(param, grad * lr)
        return params









