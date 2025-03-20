import gym
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import random
import numpy as np
from mindspore import Tensor
from mindspore.nn import Adam
from rl_utils import ReplayMemory, huber
from mindspore import load_param_into_net

class Network(nn.Cell):
    def __init__(self, len_state, num_quant, num_actions):
        super(Network, self).__init__()
        self.num_quant = num_quant
        self.num_actions = num_actions
        # 简化网络结构
        self.layer1 = nn.Dense(len_state, 256)
        self.layer2 = nn.Dense(256, num_actions * num_quant)

    def construct(self, x):
        x = self.layer1(x)
        x = ops.Tanh()(x)
        x = self.layer2(x)
        return x.view((-1, self.num_actions, self.num_quant))

    def select_action(self, state, eps):
        if not isinstance(state, Tensor):
            state = Tensor([state], dtype=mindspore.float32)
        action = Tensor([random.randint(0, 1)], dtype=mindspore.int32)
        if random.random() > eps:
            result = self.construct(state).mean(axis=2).max(axis=1)
            if len(result) > 1 and result[1].shape[0] > 0:
                action = result[1]
            else:
                action = Tensor([0], dtype=mindspore.int32)
        action = action.asnumpy()
        if action.size > 1:
            action = action[0]
        action = np.asscalar(action) if hasattr(np, 'asscalar') else action.item()
        return int(action)

def main():
    eps_start, eps_end, eps_dec = 0.9, 0.1, 200  # 加快探索率衰减
    eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)

    env_name = 'MountainCar-v0'
    # 创建环境时不进行渲染
    env = gym.make(env_name, render_mode=None)

    memory = ReplayMemory(10000)

    Z = Network(len_state=len(env.reset()[0]), num_quant=2, num_actions=env.action_space.n)
    Ztgt = Network(len_state=len(env.reset()[0]), num_quant=2, num_actions=env.action_space.n)
    optimizer = Adam(Z.trainable_params(), learning_rate=1e-4)  # 减小学习率

    steps_done = 0
    running_reward = None
    gamma, batch_size = 0.99, 32
    tau = Tensor((2 * np.arange(Z.num_quant) + 1) / (2.0 * Z.num_quant)).view((1, -1))

    all_rewards = []

    for episode in range(501):
        sum_reward = 0
        state, _ = env.reset()
        while True:
            steps_done += 1
            state_tensor = Tensor([state], dtype=mindspore.float32)
            action = Z.select_action(state_tensor, eps(steps_done))

            next_state, reward, terminated, truncated, _ = env.step(action)
            # 重新设计奖励函数
            reward += next_state[0] + 0.5
            done = terminated or truncated

            memory.push(state, action, next_state, reward, float(done))
            sum_reward += reward

            if len(memory) < batch_size:
                break
            states, actions, rewards, next_states, dones = memory.sample(batch_size)

            batch_indices = Tensor(np.arange(batch_size), dtype=mindspore.int32)
            theta = Z(states)[batch_indices, actions]

            Znext = ops.stop_gradient(Ztgt(next_states))
            batch_indices = Tensor(np.arange(batch_size), dtype=mindspore.int32)
            max_indices = Znext.mean(axis=2).max(axis=1)[1].astype(mindspore.int32)
            Znext_max = Znext[batch_indices, max_indices]
            Ttheta = rewards + gamma * (1 - dones) * Znext_max

            diff = Ttheta.transpose().expand_dims(-1) - theta
            loss = huber(diff) * ops.Abs()(tau - (ops.stop_gradient(diff) < 0).astype(mindspore.float32))
            loss = loss.mean()

            grad_fn = mindspore.ops.value_and_grad(lambda: loss, None, Z.trainable_params())
            loss, grads = grad_fn()
            optimizer(grads)

            state = next_state

            if steps_done % 100 == 0:
                params = Z.parameters_dict()
                load_param_into_net(Ztgt, params)

            if done:
                running_reward = sum_reward if not running_reward else 0.2 * sum_reward + running_reward * 0.8
                all_rewards.append(sum_reward)
                print(f"Episode {episode} finished with reward {sum_reward}")
                break

    env.close()

    print("训练结束！")
    print(f"总训练回合数: {len(all_rewards)}")
    print(f"平均奖励: {np.mean(all_rewards)}")
    print(f"最大奖励: {np.max(all_rewards)}")
    print(f"最小奖励: {np.min(all_rewards)}")

if __name__ == "__main__":
    main() 
