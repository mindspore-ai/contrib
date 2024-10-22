import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
from mindspore import Tensor, dtype as mstype
import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import value_and_grad


class MultivariateNormal:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

        # 计算协方差矩阵的 Cholesky 分解
        self.L = ops.cholesky(self.covariance)
        if isinstance(covariance, mindspore.Tensor):
            covariance = covariance.asnumpy()  # 转换为 NumPy 数组

        numpy_covariance = covariance
        numpy_inv_covariance = np.linalg.inv(numpy_covariance)
        numpy_det_covariance = np.linalg.det(numpy_covariance)
        self.inv_covariance = Tensor(numpy_inv_covariance, dtype=mstype.float32)
        self.det_covariance = Tensor(numpy_det_covariance, dtype=mstype.float32)

    def sample(self, num_samples):
        d = self.mean.shape[0]
        z = ops.randn(num_samples, d)  # 形状为 (num_samples, d) 的标准正态分布样本
        # samples = ops.matmul(z, ops.transpose(self.L, (1, 0))) + self.mean
        samples = z @ self.L.T + self.mean  # (num_samples, d) @ (d, d) + (1, d)
        return samples

    def log_prob(self, value):
        value = Tensor(value, dtype=mstype.float32)
        d = self.mean.shape[0]
        diff = value - self.mean
        if diff.dim() == 1:
            diff = diff.unsqueeze(0)  # 将其变为 (1, d)
        exponent = -0.5 * (ops.sum(diff @ self.inv_covariance * diff, dim=1))  # (num_samples, d)
        log_prob_density = exponent - 0.5 * (
                    d * ops.log(Tensor(2 * math.pi)) + ops.log(self.det_covariance))  # (num_samples, )
        return log_prob_density


class Net(nn.Cell):
    def __init__(self, n_params, num_samples):
        super().__init__()
        self.N = n_params
        self.a = nn.Dense(1, self.N, has_bias=False)
        self.b = nn.Dense(1, self.N, has_bias=False)
        self.a.weight.set_data(ops.exp(self.a.weight.data))
        self.b.weight.set_data(ops.abs(self.b.weight.data))
        self.pz = MultivariateNormal(ops.zeros(self.N), ops.eye(self.N))  # latent distribution
        self.px = MultivariateNormal(ops.zeros(self.N) + 4, 3 * ops.eye(self.N))  # target distribution
        self.num_samples = num_samples

    def construct(self, data):
        log_jacob = -ops.log(self.a.weight).sum()  # log of the determinant
        inverse = ((data - self.b.weight) / self.a.weight).mean(1)  # the inverse of the T transformation

        # computes loss
        loss = 0
        for i in range(self.N):
            log_pz = self.pz.log_prob(inverse[i])  # log of the probability
            loss += -(log_pz + log_jacob)
        loss = (1 / self.N) * loss
        return loss

    def sample(self, num_samples=100):
        t = self.px.sample(num_samples)
        og = self.pz.sample(num_samples)
        z = self.a.weight.view(-1) * og + self.b.weight.view(-1)
        return z.asnumpy(), t.asnumpy(), og.asnumpy()


if __name__ == '__main__':
    # %%
    epochs = 1000
    # %% 2 dimensional case
    net2d = Net(n_params=2, num_samples=100_000)
    optimizer = nn.Adam(net2d.trainable_params(), learning_rate=0.01)
    losses = []

    data = net2d.px.sample(net2d.num_samples).reshape(net2d.N, -1)  # sample from target distribution


    def forward_fn(inputs):
        loss = net2d(inputs)
        return loss


    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters)

    for epoch in tqdm(range(epochs)):
        (loss,), grads = grad_fn(data)  # get values and gradients
        optimizer(grads)  # update gradient
        losses.append(float(loss.asnumpy()))
        if loss < 1e-8:
            print(f'ended at epoch: {epoch}')
            break
        if (epoch + 1) % 250 == 0:
            print('loss:', round(float(loss), 3))

    losses2d = losses
    data2d = net2d.sample(200)

    modified = {
        ('latent', 'x'): data2d[0][:, 0], ('latent', 'y'): data2d[0][:, 1],
        ('target', 'x'): data2d[1][:, 0], ('target', 'y'): data2d[1][:, 1],
        ('original', 'x'): data2d[2][:, 0], ('original', 'y'): data2d[2][:, 1]
    }

    df2d = pd.DataFrame(modified)

    sns.scatterplot(data=df2d['latent'], x='x', y='y', label='latent')
    sns.scatterplot(data=df2d['target'], x='x', y='y', label='target')
    sns.scatterplot(data=df2d['original'], x='x', y='y', label='original')

    plt.savefig(fname='./2d-gaussian-approx.png')

    plt.clf()  # 清空画布
    # %% 1 dimensional case
    net1d = Net(n_params=1, num_samples=1_000)

    optimizer = nn.Adam(net1d.trainable_params(), learning_rate=0.01)
    losses = []

    data = net1d.px.sample(net1d.num_samples).reshape(net1d.N, -1)  # sample from target distribution


    def forward_fn(inputs):
        loss = net1d(inputs)
        return loss


    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters)

    for epoch in tqdm(range(epochs)):
        (loss,), grads = grad_fn(data)  # get values and gradients
        optimizer(grads)  # update gradient
        losses.append(float(loss.asnumpy()))
        if loss < 1e-8:
            print(f'ended at epoch: {epoch}')
            break
        if (epoch + 1) % 250 == 0:
            print('loss:', round(float(loss), 3))

    losses1d = losses
    data1d = net1d.sample(400)

    df1d = pd.DataFrame(
        {'latent': data1d[0].reshape(-1), 'target': data1d[1].reshape(-1), 'original': data1d[2].reshape(-1)})
    sns.histplot(data=df1d, bins='auto')
    plt.xlim([-10, 10])
    plt.savefig(fname='./1d-gaussian-approx.png')  # 保存损失图像
    plt.plot([i for i in range(len(losses1d))], losses1d, label='losses')
    plt.plot([i for i in range(len(losses2d))], losses2d, label='losses2d')
