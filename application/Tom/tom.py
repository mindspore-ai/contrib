import math
import mindspore
from mindspore import nn, Tensor
import mindspore.ops as ops


class Tom(nn.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, use_bias_correction_for_level_trend=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not (0.0 <= betas[2] < 1.0):
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        params = list(params)
        super(Tom, self).__init__(parameters=params, learning_rate=lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.use_bias_correction_for_level_trend = use_bias_correction_for_level_trend
        self.state = {}

    def construct(self, gradients):
        for param, grad in zip(self.parameters, gradients):
            if grad is None:
                continue
            if param not in self.state:
                self.state[param] = {}
                self.state[param]['step'] = 0
                self.state[param]['exp_avg'] = ops.ZerosLike()(param)
                self.state[param]['exp_trend_avg'] = ops.ZerosLike()(param)
                self.state[param]['exp_avg_sq'] = ops.ZerosLike()(param)
                self.state[param]['previous_grad'] = ops.ZerosLike()(param)
            state = self.state[param]
            beta1, beta2, beta3 = self.betas
            state['step'] += 1
            step = state['step']
            if self.use_bias_correction_for_level_trend:
                bias_corr1 = 1 - ((beta1 * beta2) ** step)
            else:
                bias_corr1 = 1
            bias_corr3 = 1 - (beta3 ** step)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param
            exp_avg = state['exp_avg']
            exp_trend_avg = state['exp_trend_avg']
            exp_avg = (exp_avg + exp_trend_avg) * beta1 + grad * (1 - beta1)
            state['exp_avg'] = exp_avg
            previous_grad = state['previous_grad']
            exp_trend_avg = exp_trend_avg * beta2 + (grad - previous_grad) * (1 - beta2)
            state['exp_trend_avg'] = exp_trend_avg
            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq = exp_avg_sq * beta3 + (1 - beta3) * (grad * grad)
            state['exp_avg_sq'] = exp_avg_sq
            total = (exp_avg + exp_trend_avg) / bias_corr1
            sqrt_exp_avg_sq = ops.Sqrt()(exp_avg_sq)
            total = total / (sqrt_exp_avg_sq / math.sqrt(bias_corr3) + self.eps)
            state['previous_grad'] = grad
            new_param = param - self.learning_rate * total
            param.set_data(new_param)
        return True


if __name__ == "__main__":
    import numpy as np
    class SimpleNet(nn.Cell):
        def __init__(self, input_dim, output_dim):
            super(SimpleNet, self).__init__()
            self.fc = nn.Dense(input_dim, output_dim)

        def construct(self, x):
            return self.fc(x)
    loss_fn = nn.MSELoss()
    net = SimpleNet(input_dim=10, output_dim=1)
    net_with_loss = nn.WithLossCell(net, loss_fn)
    optimizer = Tom(net.trainable_params(), lr=1e-3, betas=(0.9, 0.99, 0.999), eps=1e-8,
                    weight_decay=0, use_bias_correction_for_level_trend=True)
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
    print("Tom optimizer 训练步损失值：", loss_val)
