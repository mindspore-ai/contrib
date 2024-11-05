import mindspore as ms
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer

class Lamb(nn.Optimizer):
    """
    实现LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化算法。
    参数：
        params (list): 待优化的参数列表。
        learning_rate (float): 学习率，默认值为1e-3。
        beta1 (float): 一阶矩估计的指数衰减率，默认值为0.9。
        beta2 (float): 二阶矩估计的指数衰减率，默认值为0.999。
        eps (float): 为提高数值稳定性添加到分母的项，默认值为1e-6。
        weight_decay (float): 权重衰减（L2正则化），默认值为0。
    """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(Lamb, self).__init__(learning_rate, params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # 初始化状态变量
        self.moments1 = self.parameters.clone(prefix="moments1", init='zeros')
        self.moments2 = self.parameters.clone(prefix="moments2", init='zeros')
        self.steps = Parameter(initializer(0, [1], ms.int32), name="steps")

    def construct(self, gradients):
        lr = self.get_lr()
        updated_params = []

        self.steps += 1
        beta1_t = self.beta1 ** self.steps
        beta2_t = self.beta2 ** self.steps

        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            if grad is None:
                continue

            m = self.moments1[i]
            v = self.moments2[i]

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * ops.square(grad)

            m_hat = m / (1 - beta1_t)
            v_hat = v / (1 - beta2_t)

            adam_step = m_hat / (ops.sqrt(v_hat) + self.eps)

            if self.weight_decay != 0:
                adam_step += self.weight_decay * param

            weight_norm = ops.norm(param)
            adam_norm = ops.norm(adam_step)
            trust_ratio = ops.select(weight_norm > 0, ops.select(adam_norm > 0, weight_norm / adam_norm, 1.0), 1.0)

            new_param = param - lr * trust_ratio * adam_step
            updated_params.append(new_param)

            ops.assign(param, new_param)
            ops.assign(self.moments1[i], m)
            ops.assign(self.moments2[i], v)

        return updated_params