import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import numpy as np

class KneeLRScheduler:

    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=0, total_steps=0):

        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1

        assert self.decay_steps >= 0

        self.optimizer.learning_rate = Parameter(Tensor(self.get_lr(self.current_step), ms.float32), name="lr")

        if not isinstance(optimizer, nn.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        
    def get_lr(self, global_step):
        if global_step <= self.warmup_steps:
            return self.peak_lr * global_step / self.warmup_steps
        elif global_step <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(0.0, self.peak_lr + slope * (global_step - (self.explore_steps + self.warmup_steps)))


    def step(self):
        self.current_step += 1
        new_lr = self.get_lr(self.current_step)
        self.update_optimizer_lr(new_lr)

    def update_optimizer_lr(self, lr):
        self.optimizer.learning_rate = Parameter(Tensor(lr, ms.float32), name="lr")

def main():

    model = nn.Dense(10, 2)

    optimizer = nn.SGD(params=model.trainable_params(), learning_rate=Tensor(0.1, ms.float32))

    peak_lr = 0.1
    warmup_steps = 5
    explore_steps = 10
    total_steps = 50

    scheduler = KneeLRScheduler(optimizer, peak_lr, warmup_steps, explore_steps, total_steps)

    loss_fn = nn.MSELoss()

    train_step = nn.TrainOneStepCell(loss_fn, optimizer)
    train_step.set_train()

    for step in range(total_steps):
        inputs = Tensor(np.random.randn(4, 10), ms.float32)
        labels = Tensor(np.random.randn(4, 2), ms.float32)

        loss = train_step(model(inputs), labels)

        scheduler.step()
        
        current_lr = scheduler.get_lr(scheduler.current_step)
        print(f"Step {step + 1}/{total_steps}, Loss: {loss.asnumpy():.4f}, Learning Rate: {current_lr:.6f}")


if __name__ == "__main__":
    main()