import numpy as np
import mindspore as ms
from mindspore import nn, Parameter


def annealing_cos(start, end, pct: float):
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycleCosine:
    def __init__(self, optimizer: nn.Optimizer, num_steps: int,
                 warmup: float = 0.3, plateau: float = 0.,
                 winddown: float = 0.7, lr_range=(1e-5, 1e-3),
                 momentum_range=(0.85, 0.95)):
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.lr_range = lr_range
        self.momentum_range = momentum_range

        self.warmup = warmup / (warmup + plateau + winddown)
        self.plateau = plateau / (warmup + plateau + winddown)
        self.winddown = winddown / (warmup + plateau + winddown)

        self.lr_param = Parameter(ms.Tensor(lr_range[0], dtype=ms.float32),
                                  name='learning_rate')
        self.optimizer.learning_rate = self.lr_param

        self.cur_step = -1
        self.step()

    @property
    def lr(self):
        return self.lr_param.asnumpy().item()

    @lr.setter
    def lr(self, lr):
        self.lr_param.set_data(ms.Tensor(lr, dtype=ms.float32))

    @property
    def momentum(self):
        return self.optimizer.beta1

    @momentum.setter
    def momentum(self, m):
        self.optimizer.beta1 = m

    def step(self):
        self.cur_step += 1

        if 0 <= self.cur_step < self.warmup * self.num_steps:
            steps = self.cur_step
            phase_steps = self.warmup * self.num_steps
            lr = annealing_cos(self.lr_range[0], self.lr_range[1],
                              steps / phase_steps)
            momentum = annealing_cos(self.momentum_range[1],
                                    self.momentum_range[0], steps / phase_steps)
        elif (self.warmup * self.num_steps <= self.cur_step <
              (self.warmup + self.plateau) * self.num_steps):
            lr = self.lr_range[1]
            momentum = self.momentum_range[0]
        elif ((self.warmup + self.plateau) * self.num_steps <= self.cur_step <
              self.num_steps):
            steps = self.cur_step - (self.warmup + self.plateau) * self.num_steps
            phase_steps = self.winddown * self.num_steps
            lr = annealing_cos(self.lr_range[1], self.lr_range[1] / 24e4,
                              steps / phase_steps)
            momentum = annealing_cos(self.momentum_range[0],
                                    self.momentum_range[1], steps / phase_steps)
        else:
            return

        self.lr = lr
        self.momentum = momentum


class OneCycleCosineAdam(OneCycleCosine):
    @property
    def momentum(self):
        return self.optimizer.beta1

    @momentum.setter
    def momentum(self, m):
        self.optimizer.beta1 = m