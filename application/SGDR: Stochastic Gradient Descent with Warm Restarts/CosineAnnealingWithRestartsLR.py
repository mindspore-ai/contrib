import math
import matplotlib.pyplot as plt


class CosineAnnealingWithRestartsLR:
    def __init__(self, T_max, eta_min=0, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.next_restart = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.last_restart = 0

    def get_lr(self, base_lrs, cur_epoch):
        self.t_cur = cur_epoch - self.last_restart
        if self.t_cur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = cur_epoch
        return [
            (
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * self.t_cur / self.next_restart))
                    / 2
            )
            for base_lr in base_lrs
        ]


# 示例用法和绘图
if __name__ == "__main__":
    scheduler = CosineAnnealingWithRestartsLR(T_max=30, T_mult=1.5)
    lrs = []
    for step in range(240):
        lr = scheduler.get_lr([1e-3], step)
        lrs.append(lr)

    plt.plot(lrs)
    plt.savefig('cosine_annealing_with_restarts_lr.png')
