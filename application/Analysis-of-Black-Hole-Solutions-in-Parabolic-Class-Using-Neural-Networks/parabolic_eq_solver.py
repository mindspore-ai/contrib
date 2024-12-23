import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.common.initializer import XavierUniform
from mindspore.common import dtype as mstype
from mindspore import set_seed

set_seed(42)
plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class Swish(nn.Cell):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)

class FCNN(nn.Cell):
    def __init__(self, n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16), prefix=''):
        super(FCNN, self).__init__()
        self.prefix = prefix
        self.layer_names = []
        in_features = n_input_units
        for i, hidden in enumerate(hidden_units):
            dense = nn.Dense(in_features, hidden, weight_init=XavierUniform(), has_bias=True)
            dense_name = f'dense_{prefix}_{i}'
            self.__setattr__(dense_name, dense)
            self.layer_names.append(dense_name)
            swish = Swish()
            swish_name = f'swish_{prefix}_{i}'
            self.__setattr__(swish_name, swish)
            self.layer_names.append(swish_name)
            in_features = hidden
        final_dense = nn.Dense(in_features, n_output_units, weight_init=XavierUniform(), has_bias=True)
        final_dense_name = f'dense_{prefix}_{len(hidden_units)}'
        self.__setattr__(final_dense_name, final_dense)
        self.layer_names.append(final_dense_name)

    def construct(self, x):
        for name in self.layer_names:
            layer = getattr(self, name)
            x = layer(x)
        return x

class DifferentialOperator:
    def __init__(self):
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)

    def first_order(self, network, t):
        grads = self.grad(network, network.trainable_params())(t)
        return grads

    def second_order(self, network, t):
        def first_fn(t_input):
            return network(t_input)

        grad_first = self.grad(first_fn, network.trainable_params())(t)
        def second_fn(t_input):
            return grad_first[0]

        grad_second = self.grad(second_fn, network.trainable_params())(t)
        return grad_second

s0 = 100
Nepoch0 = 3000
Nt0 = 1000
Upper0 = 3
Lower0 = 0.5
w = 1.65
exp_path = 'results/experiment1/'

os.makedirs(exp_path, exist_ok=True)

b0 = np.zeros((Nt0, s0))
fm = np.zeros((Nt0, s0))
fa = np.zeros((Nt0, s0))

b0_df = pd.DataFrame(b0)
fm_df = pd.DataFrame(fm)
fa_df = pd.DataFrame(fa)

train_ls = np.zeros((Nepoch0, s0))
val_ls = np.zeros((Nepoch0, s0))

df_train_ls = pd.DataFrame(train_ls)
df_val_ls = pd.DataFrame(val_ls)

d = 5

def pbl_ode_system(b0_net, u_net, v_net, t, diff_op):
    b0 = b0_net(t)        # [Nt0, 1]
    u = u_net(t)          # [Nt0, 1]
    v = v_net(t)          # [Nt0, 1]

    eq1 = Tensor(np.zeros((Nt0, 1), dtype=np.float32))
    eq2 = Tensor(np.zeros((Nt0, 1), dtype=np.float32))
    eq3 = Tensor(np.zeros((Nt0, 1), dtype=np.float32))

    return eq1, eq2, eq3

class InitialConditions:
    def __init__(self, t0=0.0, b0_0=1.0, u0=0.0, u0_prime=0.0, fa0=0.321, fa0_prime=0.0):
        self.t0 = Tensor([[t0]], dtype=mstype.float32)          # [1, 1]
        self.b0_0 = Tensor([[b0_0]], dtype=mstype.float32)      # [1, 1]
        self.u0 = Tensor([[u0]], dtype=mstype.float32)          # [1, 1]
        self.u0_prime = Tensor([[u0_prime]], dtype=mstype.float32)  # [1, 1]
        self.fa0 = Tensor([[fa0]], dtype=mstype.float32)        # [1, 1]
        self.fa0_prime = Tensor([[fa0_prime]], dtype=mstype.float32)  # [1, 1]

ts = np.linspace(Lower0, Upper0, Nt0).astype(np.float32)
ts_tensor = Tensor(ts.reshape(-1, 1))  # 转换为列向量，形状 [Nt0, 1]

diff_op = DifferentialOperator()

class ODELoss(nn.Cell):
    def __init__(self, b0_net, u_net, v_net, condition, diff_op):
        super(ODELoss, self).__init__()
        self.b0_net = b0_net
        self.u_net = u_net
        self.v_net = v_net
        self.condition = condition
        self.diff_op = diff_op
        self.mse = nn.MSELoss()

    def construct(self, t):
        eq1, eq2, eq3 = pbl_ode_system(self.b0_net, self.u_net, self.v_net, t, self.diff_op)
        residual = eq1 ** 2 + eq2 ** 2 + eq3 ** 2

        b0_pred = self.b0_net(self.condition.t0)      # [1, 1]
        u_pred = self.u_net(self.condition.t0)        # [1, 1]
        u_prime_pred = Tensor([[0.0]], dtype=mstype.float32)  # [1, 1]
        fa_pred = self.v_net(self.condition.t0)       # [1, 1]
        fa_prime_pred = Tensor([[0.0]], dtype=mstype.float32)  # [1, 1]

        loss_ic = self.mse(b0_pred, self.condition.b0_0) \
                  + self.mse(u_pred, self.condition.u0) \
                  + self.mse(u_prime_pred, self.condition.u0_prime) \
                  + self.mse(fa_pred, self.condition.fa0) \
                  + self.mse(fa_prime_pred, self.condition.fa0_prime)

        loss = residual.mean() + loss_ic
        return loss

condition = InitialConditions()

grad_op = ops.GradOperation(get_by_list=True, sens_param=False)

for i in range(s0):
    print(f"开始模拟 {i + 1}/{s0}")

    b0_net = FCNN(n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16), prefix=f'b0_{i}')
    u_net = FCNN(n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16), prefix=f'u_{i}')
    v_net = FCNN(n_input_units=1, n_output_units=1, hidden_units=(16, 16, 16, 16), prefix=f'v_{i}')

    ode_loss = ODELoss(b0_net, u_net, v_net, condition, diff_op)
    params = list(b0_net.trainable_params()) + list(u_net.trainable_params()) + list(v_net.trainable_params())

    optimizer = nn.Adam(params, learning_rate=1e-3)

    def train_step(network, optimizer, t):
        loss = network(t)
        grads = grad_op(network, params)(t)
        optimizer(grads)
        return loss

    for epoch in range(Nepoch0):
        loss = train_step(ode_loss, optimizer, ts_tensor)
        train_ls[epoch, i] = loss.asnumpy()

        val_ls[epoch, i] = loss.asnumpy()

        if (epoch + 1) % 500 == 0:
            print(f"模拟 {i + 1}/{s0}, 训练轮数 {epoch + 1}/{Nepoch0}, 损失: {loss.asnumpy()}")

    b0_sol = b0_net(ts_tensor).asnumpy().flatten()
    fm_sol = u_net(ts_tensor).asnumpy().flatten()
    fa_sol = v_net(ts_tensor).asnumpy().flatten()

    df_train_ls.iloc[:, i] = train_ls[:, i]
    df_val_ls.iloc[:, i] = val_ls[:, i]

    b0_df.iloc[:, i] = b0_sol
    fm_df.iloc[:, i] = fm_sol
    fa_df.iloc[:, i] = fa_sol

    plt.figure()
    plt.plot(train_ls[:, i], label='训练损失')
    plt.plot(val_ls[:, i], label='验证损失')
    plt.yscale('log')
    plt.title('训练过程中的损失')
    plt.legend()
    plt.savefig(f"{exp_path}loss_w_{w}_exp_{i}.png", bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(ts, b0_sol, label='$b_0$', color='green', linewidth=1)
    plt.plot(ts, fm_sol, label='$fm$', color='red', linewidth=1)
    plt.plot(ts, fa_sol, label='$fa$', color='blue', linewidth=1)
    plt.title('当前模拟的解')
    plt.legend()
    plt.savefig(f"{exp_path}solution_w_{w}_exp_{i}.png", bbox_inches='tight')
    plt.close()

np.savetxt(f"{exp_path}ts_pbl4.txt", ts)
b0_df.to_csv(f"{exp_path}b0_pbl4_df.csv", index=False)
fm_df.to_csv(f"{exp_path}fm_pbl4_df.csv", index=False)
fa_df.to_csv(f"{exp_path}fa_pbl4_df.csv", index=False)
df_val_ls.to_csv(f"{exp_path}val_pbl4_ls.csv", index=True)
df_train_ls.to_csv(f"{exp_path}train_pbl4_ls.csv", index=True)
b0_df_net = b0_df.transpose()
fm_df_net = fm_df.transpose()
fa_df_net = fa_df.transpose()
b0_res = b0_df_net.quantile([0.025, 0.50, 0.975], axis=0)
fm_res = fm_df_net.quantile([0.025, 0.50, 0.975], axis=0)
fa_res = fa_df_net.quantile([0.025, 0.50, 0.975], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(ts, b0_res.iloc[0], label='下限 $b_0$', color='green', linewidth=1, linestyle='dashed')
plt.plot(ts, b0_res.iloc[1], label='中位数 $b_0$', color='green', linewidth=1)
plt.plot(ts, b0_res.iloc[2], label='上限 $b_0$', color='green', linewidth=1, linestyle='dashed')

plt.plot(ts, fm_res.iloc[0], label="下限 $fm$", color='blue', linewidth=1, linestyle='dashed')
plt.plot(ts, fm_res.iloc[1], label="中位数 $fm$", color='blue', linewidth=1)
plt.plot(ts, fm_res.iloc[2], label="上限 $fm$", color='blue', linewidth=1, linestyle='dashed')

plt.plot(ts, fa_res.iloc[0], label='下限 $fa$', color='red', linewidth=1, linestyle='dashed')
plt.plot(ts, fa_res.iloc[1], label='中位数 $fa$', color='red', linewidth=1)
plt.plot(ts, fa_res.iloc[2], label='上限 $fa$', color='red', linewidth=1, linestyle='dashed')

plt.ylabel('临界解')
plt.xlabel('z')
plt.title('基于ANN的抛物函数估计')
plt.legend()
plt.savefig(f"{exp_path}statistic_analysis_parabolic.pdf", bbox_inches='tight')
plt.close()

print("所有模拟完成并保存结果。")
