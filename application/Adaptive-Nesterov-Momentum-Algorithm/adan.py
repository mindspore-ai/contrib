import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor
import numpy as np


def exists(val):
    return val is not None

class Adan(nn.Optimizer):
    def __init__(
            self,
            params,
            learning_rate=1e-3,
            betas=(0.02, 0.08, 0.01),
            eps=1e-8,
            weight_decay=0,
            restart_cond=None
    ):
        assert len(betas) == 3
        super(Adan, self).__init__(learning_rate, params, weight_decay=weight_decay)

        self.beta1, self.beta2, self.beta3 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.restart_cond = restart_cond

        self.momentum = self.parameters.clone(prefix="m", init="zeros")
        self.velocity = self.parameters.clone(prefix="v", init="zeros")
        self.norm = self.parameters.clone(prefix="n", init="zeros")
        self.prev_grad = self.parameters.clone(prefix="prev_grad", init="zeros")
        self.step_count = Parameter(ms.Tensor(0, ms.int32))

    def construct(self, grads):
        lr = self.get_lr()
        self.step_count += 1

        for param, grad, m, v, n, prev_grad in zip(self.parameters, grads, self.momentum, self.velocity, self.norm,
                                                   self.prev_grad):
            if not exists(grad):
                continue

            grad_diff = grad - prev_grad
            m = (1 - self.beta1) * m + self.beta1 * grad
            v = (1 - self.beta2) * v + self.beta2 * grad_diff
            next_n = (grad + (1 - self.beta2) * grad_diff) ** 2
            n = (1 - self.beta3) * n + self.beta3 * next_n

            correct_m = 1 / (1 - (1 - self.beta1) ** self.step_count)
            correct_v = 1 / (1 - (1 - self.beta2) ** self.step_count)
            correct_n = 1 / (1 - (1 - self.beta3) ** self.step_count)

            weighted_step_size = lr / (ops.sqrt(n * correct_n) + self.eps)
            denom = 1 + self.weight_decay * lr
            update = weighted_step_size * (m * correct_m + (1 - self.beta2) * v * correct_v)
            ops.assign(param, (param - update) / denom)

            ops.assign(prev_grad, grad)

            if exists(self.restart_cond) and self.restart_cond(self.step_count):
                ops.assign(m, grad)
                ops.assign(v, ops.zeros_like(v))
                ops.assign(n, grad ** 2)

        return True

if __name__ == '__main__':
    class SimpleNet(nn.Cell):
        def __init__(self, input_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc = nn.Dense(input_size, output_size)

        def construct(self, x):
            return self.fc(x)


    np.random.seed(0)
    input_data = np.random.randn(100, 1).astype(np.float32)
    labels = 2 * input_data + 1
    input_tensor = Tensor(input_data)
    label_tensor = Tensor(labels)
    net = SimpleNet(input_size=1, output_size=1)
    loss_fn = nn.MSELoss()
    optimizer = Adan(params=net.trainable_params())

    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    num_epochs = 100

    for epoch in range(num_epochs):
        (loss, _), grads = grad_fn(input_tensor, label_tensor)
        optimizer(grads)
        print(f"Epoch {epoch + 1}, Loss: {loss.asnumpy()}")