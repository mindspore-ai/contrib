import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mindspore import nn, ops, Tensor
import mindspore.numpy as mnp

from network_mindspore import RealNVP
from loss_mindspore import Loss


noisy_moons, label = datasets.make_moons(n_samples=1000, noise=0.05)

model = RealNVP()
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)

# 定义多元正态分布，用于计算 log_prob
class MultivariateNormal:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dim = mean.shape[0]
        self.mat_inv = ops.MatrixInverse()
        self.mat_det = ops.MatrixDeterminant()
        self.cov_inv = self.mat_inv(cov)
        self.log_det_cov = mnp.log(self.mat_det(cov))
        self.const = -0.5 * self.dim * mnp.log(2 * mnp.pi)
    
    def log_prob(self, x):
        diff = x - self.mean
        quad = mnp.sum(mnp.matmul(diff, self.cov_inv) * diff, axis=1)
        return self.const - 0.5 * (self.log_det_cov + quad)
    
mean = Tensor(np.zeros(2, dtype=np.float32))
cov = Tensor(np.eye(2, dtype=np.float32))
prior = MultivariateNormal(mean, cov)

loss_log_det_jacobians = Loss(prior)


#TRAIN

epochs = 1000

# 定义包含损失计算的封装网络
class WithLossCell(nn.Cell):
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__()
        self.net = net
        self.loss_fn = loss_fn

    def construct(self, x):
        z, sum_log_det_jacobian = self.net(x)
        loss = self.loss_fn(z, sum_log_det_jacobian)
        return loss

net_with_loss = WithLossCell(model, loss_log_det_jacobians)
train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
train_network.set_train()

for epoch in range(epochs):
    x_np = datasets.make_moons(n_samples=1000, noise=0.05)[0].astype(np.float32)
    x = Tensor(x_np)
    
    loss = train_network(x)
    
    if epoch % 200 == 0:
        print('(epoch {}/{}) loss : {:.3f}'.format(epoch, epochs, loss.asnumpy()))
        # test
        model.set_train(False)
        x_test = Tensor(noisy_moons.astype(np.float32))
        z, _ = model(x_test)
        z_np = z.asnumpy()
        plt.scatter(z_np[:, 0], z_np[:, 1])
        plt.show()


# TEST

# Inference (x -> z)
x, label = datasets.make_moons(n_samples=1000, noise=0.05)
x = x.astype(np.float32)
plt.scatter(x[:, 0], x[:, 1], c=label)
plt.title("Original Data")
plt.show()

z, _ = model(Tensor(x))
z_np = z.asnumpy()
plt.scatter(z_np[:, 0], z_np[:, 1], c=label)
plt.title("Transformed Data")
plt.show()


# Generate (z -> x)
z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000).astype(np.float32)
plt.scatter(z[:, 0], z[:, 1])
plt.title("Original Latent Samples")
plt.show()

x = model(Tensor(z), reverse=True)
x_np = x.asnumpy()
plt.scatter(x_np[:, 0], x_np[:, 1])
plt.title("Transformed Data")
plt.show()
