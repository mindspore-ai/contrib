import mindspore as ms
from mindspore import Tensor, ops, context
import numpy as np

# 设置MindSpore运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class SVGD_model():
    def __init__(self):
        pass

    def SVGD_kernel(self, x, h=-1):
        # 计算pairwise距离矩阵
        x_norm = ops.ReduceSum(keep_dims=True)(x ** 2, axis=1)
        pairwise_dists = x_norm - 2 * ops.matmul(x, x.T) + x_norm.T

        if h < 0:  # 使用中值技巧
            pairwise_dists_np = pairwise_dists.asnumpy()
            h_value = np.median(pairwise_dists_np)
            h = Tensor(h_value ** 2 / np.log(x.shape[0] + 1), ms.float32)
        else:
            h = Tensor(h, ms.float32)

        kernel_xj_xi = ops.exp(-pairwise_dists ** 2 / h)

        # 计算核函数的梯度
        x_expand = x[:, None, :]  # 扩展维度 (n_samples, 1, n_features)
        x_diff = x_expand - x[None, :, :]  # 差值矩阵 (n_samples, n_samples, n_features)
        weights = kernel_xj_xi[:, :, None]  # 权重矩阵 (n_samples, n_samples, 1)
        weighted_diff = weights * x_diff
        d_kernel_xi = ops.ReduceSum()(weighted_diff, axis=1) * 2 / h

        return kernel_xj_xi, d_kernel_xi

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        x = x0.copy()
        eps_factor = 1e-8
        historical_grad_square = None

        for iter in range(n_iter):
            if debug and (iter + 1) % 100 == 0:
                print('iter ' + str(iter + 1))

            kernel_xj_xi, d_kernel_xi = self.SVGD_kernel(x, h=bandwidth)
            grad_lnprob = lnprob(x)

            current_grad = (ops.matmul(kernel_xj_xi, grad_lnprob) + d_kernel_xi) / x.shape[0]

            if historical_grad_square is None:
                historical_grad_square = current_grad ** 2
            else:
                historical_grad_square = alpha * historical_grad_square + (1 - alpha) * (current_grad ** 2)

            adj_grad = current_grad / ops.sqrt(historical_grad_square + eps_factor)
            x += stepsize * adj_grad

        return x

class MVN():
    def __init__(self, mean, cov):
        self.mean = Tensor(mean, ms.float32)
        self.cov = Tensor(cov, ms.float32)
        self.cov_inv = ops.inv(self.cov)

    def dlnprob(self, x):
        diff = x - self.mean
        grad = -ops.matmul(diff, self.cov_inv)
        return grad

if __name__ == "__main__":
    # 定义均值和协方差矩阵
    cov_np = np.array([[0.2260, 0.1652], [0.1652, 0.6779]], dtype=np.float32)
    mean_np = np.array([-0.6871, 0.8010], dtype=np.float32)

    mvn_model = MVN(mean_np, cov_np)

    np.random.seed(0)
    x0_np = np.random.normal(0, 1, [10, 2]).astype(np.float32)
    x0 = Tensor(x0_np, ms.float32)

    dlnprob = mvn_model.dlnprob

    svgd_model = SVGD_model()
    x = svgd_model.update(x0, dlnprob, n_iter=1000, stepsize=1e-2, bandwidth=-1, alpha=0.9, debug=True)

    x_np = x.asnumpy()

    print("Mean ground truth: ", mean_np)
    print("Mean obtained by SVGD: ", np.mean(x_np, axis=0))

    print("Covariance ground truth: \n", cov_np)
    print("Covariance obtained by SVGD: \n", np.cov(x_np.T))
