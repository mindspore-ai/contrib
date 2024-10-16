import math
import matplotlib.pyplot as plt
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import mindspore


class CosineAnnealingWithRestartsLR(LearningRateSchedule):
    def __init__(self, T_max, eta_min=0, T_mult=1, base_lrs=[1e3]):
        super(CosineAnnealingWithRestartsLR, self).__init__()
        self.T_max = T_max
        self.T_mult = T_mult
        self.next_restart = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.last_restart = 0
        self.base_lrs = base_lrs

    def construct(self, current_step):
        self.t_cur = current_step - self.last_restart
        if self.t_cur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = current_step
        return mindspore.Tensor([
            (
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.t_cur / self.next_restart))
                / 2
            )
            for base_lr in self.base_lrs
        ])


if __name__ == "__main__":
    scheduler = CosineAnnealingWithRestartsLR(T_max=30, T_mult=1.5)
    lrs = []
    for step in range(240):
        lr = scheduler(step)
        lrs.append(lr)
    plt.plot(lrs)
    plt.savefig('cosine_annealing_with_restarts_lr.png')
