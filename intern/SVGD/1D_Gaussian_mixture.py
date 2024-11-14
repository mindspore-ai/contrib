import mindspore as ms
from mindspore import Tensor, ops, nn, context
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SVGD_model():
    def __init__(self):
        pass

    def SVGD_kernel(self, x, h=-1):
        # 计算pairwise距离矩阵
        x_norm = ops.ReduceSum(keep_dims=True)(x ** 2, axis=1)
        pairwise_dists = x_norm - 2 * ops.matmul(x, x.T) + x_norm.T

        if h < 0:  # 使用中值技巧
            h_value = np.median(pairwise_dists.asnumpy())
            h = Tensor(h_value ** 2 / np.log(x.shape[0] + 1), ms.float32)
        else:
            h = Tensor(h, ms.float32)

        kernel_xj_xi = ops.exp(-pairwise_dists / h)

        # 计算核函数的梯度
        x_expand = x[:, None, :]  # 扩展维度
        x_diff = x_expand - x[None, :, :]  # 计算差值
        weights = kernel_xj_xi[:, :, None]
        weighted_diff = weights * x_diff
        d_kernel_xi = ops.ReduceSum()(weighted_diff, axis=1) * 2 / h

        return kernel_xj_xi, d_kernel_xi

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        x = x0.copy()
        eps_factor = 1e-8
        historical_grad_square = 0

        for iter in range(n_iter):
            if debug and (iter + 1) % 100 == 0:
                print('iter ' + str(iter + 1))

            kernel_xj_xi, d_kernel_xi = self.SVGD_kernel(x, h=bandwidth)
            grad_lnprob = lnprob(x)

            current_grad = (ops.matmul(kernel_xj_xi, grad_lnprob) + d_kernel_xi) / x.shape[0]

            if iter == 0:
                historical_grad_square = current_grad ** 2
            else:
                historical_grad_square = alpha * historical_grad_square + (1 - alpha) * (current_grad ** 2)

            adj_grad = current_grad / ops.sqrt(historical_grad_square + eps_factor)
            x += stepsize * adj_grad

        return x

class OneDimensionGM():
    def __init__(self, omega, mean, var):
        self.omega = Tensor(omega, ms.float32)
        self.mean = Tensor(mean, ms.float32)
        self.var = Tensor(var, ms.float32)

    def dlnprob(self, x):
        rep_x = ops.tile(x, (1, self.omega.shape[0]))
        exponent = - (rep_x - self.mean) ** 2 / (2 * self.var)
        coef = 1 / ops.sqrt(2 * np.pi * self.var)
        category_prob = ops.exp(exponent) * coef * self.omega

        den = ops.ReduceSum()(category_prob, axis=1)
        num = ops.ReduceSum()((-(rep_x - self.mean) / self.var) * category_prob, axis=1)

        return (num / den).reshape(-1, 1)

    def MGprob(self, x):
        rep_x = ops.tile(x, (1, self.omega.shape[0]))
        exponent = - (rep_x - self.mean) ** 2 / (2 * self.var)
        coef = 1 / ops.sqrt(2 * np.pi * self.var)
        category_prob = ops.exp(exponent) * coef * self.omega

        den = ops.ReduceSum()(category_prob, axis=1)
        return den.reshape(-1, 1)

if __name__ == "__main__":
    sns.set_palette('deep', desat=.6)
    sns.set_context(rc={'figure.figsize': (8, 5)})

    w = np.array([1/3, 2/3], dtype=np.float32)
    mean = np.array([-2, 2], dtype=np.float32)
    var = np.array([1, 1], dtype=np.float32)

    OneDimensionGM_model = OneDimensionGM(w, mean, var)

    np.random.seed(0)
    x0_np = np.random.normal(-10, 1, [100, 1]).astype(np.float32)
    x0 = Tensor(x0_np, ms.float32)

    dlnprob = OneDimensionGM_model.dlnprob

    svgd_model = SVGD_model()
    n_iter = 500
    x = svgd_model.update(x0, dlnprob, n_iter=n_iter, stepsize=1e-1, bandwidth=-1, alpha=0.9, debug=True)

    # 绘制结果
    x_np = x.asnumpy().reshape(-1)
    sns.kdeplot(x_np, bw=.4, color='g')

    x_lin_np = np.linspace(-15, 15, 100).reshape(-1, 1).astype(np.float32)
    x_lin = Tensor(x_lin_np, ms.float32)
    x_prob = OneDimensionGM_model.MGprob(x_lin).asnumpy()

    plt.plot(x_lin_np, x_prob, 'b--')
    plt.axis([-15, 15, 0, 0.4])
    plt.title(f'{n_iter}$ ^{{th}}$ iteration')
    plt.show()
