from collections.abc import Iterable
from math import log, cos, pi, floor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import numpy as np
import mindspore
from mindspore import Tensor

class CyclicCosineDecayLR(LearningRateSchedule):
    def __init__(self,
                 base_lr,
                 init_decay_epochs,
                 min_decay_lr,
                 total_epochs,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr=None,
                 warmup_epochs=None,
                 warmup_start_lr=None):

        super(CyclicCosineDecayLR, self).__init__()
        
        # 首先确定参数组的数量
        if isinstance(base_lr, list):
            self.num_groups = len(base_lr)
        else:
            self.num_groups = 1
        
        # 将学习率参数统一为列表形式，以支持多参数组
        self.base_lrs = self._format_param(base_lr)
        self.min_decay_lrs = self._format_param(min_decay_lr)
        self.restart_lrs = self._format_param(restart_lr) if restart_lr is not None else self.base_lrs
        self.warmup_start_lrs = self._format_param(warmup_start_lr) if warmup_start_lr is not None else None

        self.init_decay_epochs = init_decay_epochs
        self.total_epochs = total_epochs
        self.restart_interval = restart_interval
        self.restart_interval_multiplier = restart_interval_multiplier
        self.warmup_epochs = warmup_epochs

        if warmup_epochs is not None and warmup_start_lr is None:
            raise ValueError("warmup_start_lr 必须在 warmup_epochs 不为 None 时设置")

        self.lrs = self.compute_lr_schedule()

    def construct(self, global_step):
        # 返回当前 step 的学习率列表
        if global_step >= len(self.lrs):
            global_step = len(self.lrs) - 1  # 防止索引越界
        lrs = [Tensor(self.lrs[global_step][i], mindspore.float32) for i in range(self.num_groups)]
        return lrs

    def compute_lr_schedule(self):
        lrs = []
        for epoch in range(self.total_epochs):
            lr_epoch = []
            for idx in range(self.num_groups):
                base_lr = self.base_lrs[idx]
                min_lr = self.min_decay_lrs[idx]
                restart_lr = self.restart_lrs[idx]
                warmup_start_lr = self.warmup_start_lrs[idx] if self.warmup_start_lrs is not None else None

                if self.warmup_epochs is not None and epoch < self.warmup_epochs:
                    # 预热阶段
                    t = epoch
                    T = self.warmup_epochs
                    lr = self._calc_lr(t, T, warmup_start_lr, base_lr)
                elif epoch < (self.warmup_epochs or 0) + self.init_decay_epochs:
                    # 初始衰减阶段
                    t = epoch - (self.warmup_epochs or 0)
                    T = self.init_decay_epochs
                    lr = self._calc_lr(t, T, base_lr, min_lr)
                else:
                    # 周期性阶段
                    if self.restart_interval is not None:
                        effective_epoch = epoch - (self.warmup_epochs or 0) - self.init_decay_epochs
                        if self.restart_interval_multiplier is None:
                            # 固定周期
                            T = self.restart_interval
                            cycle_epoch = effective_epoch % T
                            lr_start = restart_lr
                            lr_end = min_lr
                            lr = self._calc_lr(cycle_epoch, T, lr_start, lr_end)
                        else:
                            # 几何增长周期
                            n = self._get_n(effective_epoch)
                            sn_prev = self._partial_sum(n)
                            cycle_epoch = effective_epoch - sn_prev
                            T = self.restart_interval * (self.restart_interval_multiplier ** n)
                            lr_start = restart_lr
                            lr_end = min_lr
                            lr = self._calc_lr(cycle_epoch, T, lr_start, lr_end)
                    else:
                        # 无周期，保持最小学习率
                        lr = min_lr
                lr_epoch.append(lr)
            lrs.append(lr_epoch)
        return lrs

    def _calc_lr(self, t, T, lr_start, lr_end):
        # 余弦衰减公式
        cos_inner = np.pi * t / T
        cos_out = (1 + np.cos(cos_inner)) / 2
        lr = lr_end + (lr_start - lr_end) * cos_out
        return lr

    def _get_n(self, epoch):
        if self.restart_interval_multiplier == 1 or self.restart_interval_multiplier is None:
            n = int(epoch // self.restart_interval)
        else:
            numerator = 1 - (1 - self.restart_interval_multiplier) * epoch / self.restart_interval
            if numerator <= 0:
                return 0
            n = int(np.floor(np.log(numerator) / np.log(self.restart_interval_multiplier)))
        return n

    def _partial_sum(self, n):
        if self.restart_interval_multiplier == 1 or self.restart_interval_multiplier is None:
            return n * self.restart_interval
        else:
            return self.restart_interval * (1 - self.restart_interval_multiplier ** n) / (1 - self.restart_interval_multiplier)

    def _format_param(self, param):
        if isinstance(param, (float, int)):
            return [param] * self.num_groups
        elif isinstance(param, list):
            if len(param) != self.num_groups:
                raise ValueError()
            return param
        elif param is None:
            return [None] * self.num_groups
        else:
            raise TypeError()


