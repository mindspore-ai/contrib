import math
import mindspore
from mindspore import ops, nn
import numpy as np


class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        print(f'GX in rbf: {np.sum(GX)},\t KF in rbf: {np.sum(KX)}')
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


class MindsporeCKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = ops.ones([n, n])
        I = ops.eye(n)
        H = I - unit / n
        return ops.matmul(ops.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = ops.matmul(X, X.T)
        # FIXME: function 'diag' is not compatible
        # KX = mindspore.numpy.diag(GX) - GX + (mindspore.numpy.diag(GX) - GX).T
        KX = ops.diag(GX) - GX + (ops.diag(GX) - GX).T
        print(f'GX in rbf: {ops.sum(GX)},\t KF in rbf: {ops.sum(KX)}')
        if sigma is None:
            mdist = ops.median(KX[KX != 0])
            sigma = ops.sqrt(mdist[0])
        KX *= - 0.5 / (sigma * sigma)
        KX = ops.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return ops.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = ops.matmul(X, X.T)
        L_Y = ops.matmul(Y, Y.T)
        return ops.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = ops.sqrt(self.linear_HSIC(X, X))
        var2 = ops.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = ops.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = ops.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


if __name__ == '__main__':
    op_np = CKA()
    op_ms = MindsporeCKA()

    import time
    seed = int(time.time())
    X_ms = ops.randn([4, 2], seed=seed)
    Y_ms = ops.randn([4, 2], seed=seed)
    X_np = X_ms.asnumpy()
    Y_np = Y_ms.asnumpy()

    print('numpy:')
    # print('Linear CKA, between X and Y: {}'.format(op_np.linear_CKA(X_np, Y_np)))
    # print('Linear CKA, between X and X: {}'.format(op_np.linear_CKA(X_np, X_np)))
    print('Kernel HSIC, between X and Y: {}'.format(
        op_np.kernel_HSIC(X_np, Y_np, None)))
    # print('Kernel HSIC, between X and X: {}'.format(op_np.kernel_HSIC(X_np, X_np, None)))
    # print('Kernel HSIC, between Y and Y: {}'.format(op_np.kernel_HSIC(Y_np, Y_np, None)))
    # print('RBF Kernel CKA, between X and Y: {}'.format(op_np.kernel_CKA(X_np, Y_np)))
    # print('RBF Kernel CKA, between X and X: {}'.format(op_np.kernel_CKA(X_np, X_np)))

    print('mindspore:')
    # print('Linear CKA, between X and Y: {}'.format(op_ms.linear_CKA(X_ms, Y_ms)))
    # print('Linear CKA, between X and X: {}'.format(op_ms.linear_CKA(X_ms, X_ms)))
    print('Kernel HSIC, between X and Y: {}'.format(
        op_ms.kernel_HSIC(X_ms, Y_ms, None)))
    # print('Kernel HSIC, between X and X: {}'.format(op_ms.kernel_HSIC(X_ms, X_ms, None)))
    # print('Kernel HSIC, between Y and Y: {}'.format(op_ms.kernel_HSIC(Y_ms, Y_ms, None)))
    # print('RBF Kernel CKA, between X and Y: {}'.format(op_ms.kernel_CKA(X_ms, Y_ms)))
    # print('RBF Kernel CKA, between X and X: {}'.format(op_ms.kernel_CKA(X_ms, X_ms)))
