import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn
import numpy as np
from onecyclec_mindspore import OneCycleCosineAdam


class ToyModel(nn.Cell):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Dense(1, 1)

    def construct(self, x):
        return self.fc(x)


model = ToyModel()
optimizer = nn.AdamWeightDecay(model.trainable_params(),
                               learning_rate=0.01, beta1=0.9, beta2=0.999)

WARMUP = 0.3
PLATEAU = 0.1
WINDDOWN = 0.7
N = 1000

sched = OneCycleCosineAdam(optimizer, warmup=WARMUP, plateau=PLATEAU,
                           winddown=WINDDOWN, num_steps=N)

momentum = []
lr = []

for n in range(0, N):
    momentum.append(sched.momentum)
    lr.append(sched.lr)
    sched.step()

lr = np.array(lr)
momentum = np.array(momentum)

fig, ax1 = plt.subplots()
plt.grid()

ax2 = ax1.twinx()

ax1.set_xlabel('step')
ax1.set_ylabel('learning rate', color='tab:blue')
ax1.plot(lr, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2.set_ylabel('momentum', color='green')
ax2.plot(momentum, color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title(f'warmup = {WARMUP:.1f} plateau = {PLATEAU:.1f} winddown = {WINDDOWN:.1f}')
plt.savefig('sched.png', dpi=100, bbox_inches='tight')
plt.show()