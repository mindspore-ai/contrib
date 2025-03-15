import numpy as np
import mindspore
from mindspore import nn, Tensor, Parameter
import mindspore.ops as ops


class CSGD(nn.Optimizer):
    def __init__(self, params, learning_rate=0.1, momentum=0.0, weight_decay=0.0):
        if learning_rate < 0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        params = list(params)
        super(CSGD, self).__init__(parameters=params, learning_rate=learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum_buffers = {}
        self.steps = {}
        if self.momentum != 0:
            for param in self.parameters:
                self.momentum_buffers[param.name] = Parameter(ops.ZerosLike()(param), name=param.name + "_momentum",
                                                              requires_grad=False)
                self.steps[param.name] = 0

    def construct(self, gradients):
        for param, grad in zip(self.parameters, gradients):
            if grad is None:
                continue
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            lr = self.learning_rate
            if self.momentum != 0:
                self.steps[param.name] += 1
                step = self.steps[param.name]
                m_buf = self.momentum_buffers[param.name]
                m_buf = m_buf * self.momentum + (1 - self.momentum) * grad
                self.momentum_buffers[param.name].set_data(m_buf)
                bias_correction = 1 - self.momentum ** step
                lr = lr / bias_correction
                grad = m_buf
            param.set_data(param - lr * grad)
        return True

if __name__ == '__main__':
    class SimpleNet(nn.Cell):
        def __init__(self, input_dim, output_dim):
            super(SimpleNet, self).__init__()
            self.fc = nn.Dense(input_dim, output_dim)
        def construct(self, x):
            return self.fc(x)
    loss_fn = nn.MSELoss()
    net = SimpleNet(input_dim=10, output_dim=1)
    net_with_loss = nn.WithLossCell(net, loss_fn)
    optimizer = CSGD(net.trainable_params(), learning_rate=0.1, momentum=0.9, weight_decay=0.01)
    x_np = np.arange(50, dtype=np.float32).reshape(5, 10)
    target_np = np.arange(5, dtype=np.float32).reshape(5, 1)
    x = Tensor(x_np)
    target = Tensor(target_np)
    grad_fn = mindspore.value_and_grad(net_with_loss, None, net.trainable_params(), has_aux=False)
    def train_step(x, target):
        loss, grads = grad_fn(x, target)
        optimizer(grads)
        return loss
    loss_val = train_step(x, target)
    print("Mindspore 训练步损失值：", loss_val)
