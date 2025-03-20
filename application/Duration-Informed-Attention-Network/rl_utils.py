import mindspore
import mindspore.nn as nn
import random

def huber(x, k=1.0):
    abs_x = mindspore.ops.Abs()(x)
    return mindspore.ops.where(abs_x < k, 0.5 * x**2, k * (abs_x - 0.5 * k))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        transition = (mindspore.Tensor(state), mindspore.Tensor([action]), mindspore.Tensor(next_state), mindspore.Tensor([reward]), mindspore.Tensor([done]))
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)
        
        batch_state = mindspore.ops.stack(batch_state)
        batch_action = mindspore.Tensor(batch_action, dtype=mindspore.int32)
        batch_reward = mindspore.ops.stack(batch_reward)
        batch_done = mindspore.ops.stack(batch_done)
        batch_next_state = mindspore.ops.stack(batch_next_state)
        
        return batch_state, batch_action, batch_reward.expand_dims(1), batch_next_state, batch_done.expand_dims(1)
    
    def __len__(self):
        return len(self.memory)
