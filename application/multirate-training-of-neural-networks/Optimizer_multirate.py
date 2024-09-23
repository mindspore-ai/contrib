import numpy as np
import mindspore
from mindspore import nn,ops,Tensor,Parameter
from mindspore.common.initializer import initializer

class Multirate(nn.Cell):
    def __init__(self, params, lr=0.1, momentum=0, weight_decay=0):
        super(Multirate, self).__init__(auto_prefix=False)
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum_buffers = {}
        self.assign = ops.Assign()
        self.mul = ops.Mul()
        self.sub = ops.Sub()

    def construct(self, gradients):
        for param, grad in zip(self.params, gradients):
            if grad is None:
                continue
            if param not in self.momentum_buffers:
                self.momentum_buffers[param] = initializer('zeros', param.shape)
            buf = self.momentum_buffers[param]
            buf = self.mul(buf, self.momentum) + grad
            if self.weight_decay != 0:
                buf += self.mul(param, self.weight_decay)
            param = self.sub(param, self.mul(buf, self.lr))
            self.assign(param, param)
            self.assign(buf, buf)
        return self.params

class WithLossCell(nn.Cell):
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, data, label):
        output = self.network(data)
        return self.loss_fn(output, label)

# 定义一个简单的模型
class SimpleModel(nn.Cell):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = Parameter(initializer('normal', [10, 1]))

    def construct(self, x):
        return ops.operations.MatMul()(x, self.linear)

if __name__ == '__main__':
    model = SimpleModel()
    loss = nn.MSELoss()
    optimizer = Multirate(model.trainable_params(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    X = Tensor(np.random.randn(100, 10), dtype=mindspore.float32)
    y = Tensor(np.random.randn(100, 1), dtype=mindspore.float32)

    with_loss_cell = WithLossCell(model, loss)

    grad_ops = ops.composite.GradOperation(get_by_list=True)

    num_epochs = 5
    for epoch in range(num_epochs):
        gradients = grad_ops(with_loss_cell)(X, y)
        optimizer(gradients)
        outputs = model(X)
        loss_value = loss(outputs, y)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss_value}')

    for param in model.trainable_params():
        print(param)