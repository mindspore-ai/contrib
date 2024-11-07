import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer

class DoWG(nn.Optimizer):
    """
    实现 DoWG（Distance over Weight Gradient）优化算法。

    参数：
        parameters (list): 待优化的参数列表。
        learning_rate (float): 学习率，默认值为 0.001。
        eps (float): 增加到分母以提高数值稳定性的项，默认值为 1e-4。
    """
    def __init__(self, parameters, learning_rate=0.001, eps=1e-4):
        super(DoWG, self).__init__(learning_rate=learning_rate, parameters=parameters)
        self.eps = eps
        self.rt2 = Parameter(initializer(Tensor([eps], ms.float32), [1]), name="rt2")
        self.vt = Parameter(initializer(Tensor([0.0], ms.float32), [1]), name="vt")
        self.x0 = [Parameter(p.clone(), name=f"x0_{i}") for i, p in enumerate(parameters)]

        self.assign = ops.Assign()
        self.assign_add = ops.AssignAdd()

    def construct(self, gradients):
        grad_sq_norm = ops.zeros((1,), ms.float32)
        curr_d2 = ops.zeros((1,), ms.float32)

        for p, g, x0 in zip(self.parameters, gradients, self.x0):
            grad_sq_norm += ops.reduce_sum(g ** 2)
            curr_d2 += ops.reduce_sum((p - x0) ** 2)

        new_rt2 = ops.maximum(self.rt2, curr_d2)
        self.assign(self.rt2, new_rt2)
        self.assign_add(self.vt, self.rt2 * grad_sq_norm)

        denom = ops.sqrt(self.vt) + self.eps

        for p, g in zip(self.parameters, gradients):
            gt_hat = self.rt2 * g
            new_p = p - gt_hat / denom
            self.assign(p, new_p)

        return True

class CDoWG(nn.Optimizer):
    """
    实现 CDoWG（Coordinate-wise Distance over Weight Gradient）优化算法。
    参数：
        params (list): 待优化的参数列表。
        eps (float): 增加到分母以提高数值稳定性的项，默认值为 1e-8。
    """
    def __init__(self, params, eps=1e-8):
        super(CDoWG, self).__init__(learning_rate=0.001, params=params)
        self.eps = eps
        self.x0 = [Parameter(p.clone(), name=f"x0_{i}") for i, p in enumerate(params)]
        self.rt2 = [Parameter(initializer(1e-4, p.shape, ms.float32), name=f"rt2_{i}") for i, p in enumerate(params)]
        self.vt = [Parameter(initializer(0, p.shape, ms.float32), name=f"vt_{i}") for i, p in enumerate(params)]

    def construct(self, gradients):
        for p, g, x0, rt2, vt in zip(self.parameters, gradients, self.x0, self.rt2, self.vt):
            rt2 = ops.maximum(rt2, (p - x0) ** 2)
            vt += rt2 * g ** 2
            gt_hat = rt2 * g
            denom = ops.sqrt(vt) + self.eps
            p = p - gt_hat / denom

        return True