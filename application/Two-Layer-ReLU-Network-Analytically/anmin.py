import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as ms_np
from mindspore.dataset import GeneratorDataset
from time import time

def Fmat(A, x):
    """
    将 ReLU 激活应用于 x 和 A 的矩阵乘积，然后在零处进行阈值处理。

    参数:
        A (Tensor): 权重矩阵。
        x (Tensor): 输入张量。

    返回:
        Tensor: 阈值化后的张量。
    """
    relu = ops.ReLU()
    x1 = relu(x @ A)
    u = (x1 > 0).astype(ms.float32)
    return u

def anet1(A, b, b0, F, x):
    """
    计算线性网络层的输出。

    参数:
        A (Tensor): 权重矩阵。
        b (Tensor): 权重向量。
        b0 (Tensor): 偏置项。
        F (Tensor): 特征张量。
        x (Tensor): 输入张量。

    返回:
        Tensor: 输出张量。
    """
    Ax = x @ A
    py = (F * Ax) @ b + b0
    return py

def anet(A, b, b0, x, al):
    """
    通过组合 ReLU 激活应用线性变换。

    参数:
        A (Tensor): 权重矩阵。
        b (Tensor): 权重向量。
        b0 (Tensor): 偏置项。
        x (Tensor): 输入张量。
        al (float): 激活参数。

    返回:
        Tensor: 输出张量。
    """
    xa = x @ A
    Ax = ops.relu(xa) * (1 - al) + xa * al
    py = Ax @ b + b0
    return py

def fitOLSw(x, y, w, la=0.0001):
    """
    使用加权最小二乘法拟合模型。

    参数:
        x (Tensor): 输入特征。
        y (Tensor): 目标值。
        w (Tensor): 权重。
        la (float, 可选): 正则化参数，默认值为 0.0001。

    返回:
        Tuple[Tensor, Tensor]: 拟合的系数和截距。
    """
    ones = ops.ones((x.shape[0], 1), dtype=x.dtype)
    x1 = ops.concat((ones, x), axis=1)
    w_view = ops.reshape(w, (-1, 1))
    xx = ops.matmul(x1.transpose(), x1 * w_view)
    xy = ops.matmul(x1.transpose(), y * w_view)
    eye = ops.eye(xx.shape[0], xx.shape[1], dtype=xx.dtype)
    A_matrix = xx + eye * la
    b_sol = ms_np.linalg.solve(A_matrix, xy)
    if len(b_sol.shape) == 1:
        return b_sol[1:], b_sol[0]
    else:
        return b_sol[1:, :], b_sol[0, :]

def fitAw(F, x, y, w, b, b0, la=0.0001):
    """
    拟合矩阵 A，以最小化加权损失 ||U*(xA)b + b0 - y||^2。

    参数:
        F (Tensor): 特征矩阵。
        x (Tensor): 输入特征。
        y (Tensor): 目标值。
        w (Tensor): 权重。
        b (Tensor): 权重向量。
        b0 (Tensor): 偏置项。
        la (float, 可选): 正则化参数。默认值为 0.0001。

    返回:
        Tensor: 拟合的矩阵 A。
    """
    x_np = x.asnumpy()
    F_np = F.asnumpy()
    w_np = w.asnumpy()
    y_np = y.asnumpy()

    def data_generator():
        for i in range(x_np.shape[0]):
            yield (x_np[i], F_np[i], w_np[i], y_np[i])

    dataset = GeneratorDataset(source=data_generator, column_names=["x", "F", "w", "y"])
    data_loader = dataset.batch(8192, drop_remainder=False).shuffle(buffer_size=10000)

    yb0 = y - b0
    xy = w.reshape(-1, 1) * yb0 * x
    uxy = ops.matmul(F.T, xy)
    buxy = uxy * b
    buxy = buxy.reshape(-1, 1)

    d = x.shape[1]
    h = F.shape[1]
    dh = d * h

    xx = np.zeros((h, h, d, d), dtype=x_np.dtype)

    for batch in data_loader.create_dict_iterator():
        xi = batch["x"].asnumpy()
        fi = batch["F"].asnumpy()
        wi = batch["w"].asnumpy()
        xu = np.einsum('bi,bj->bij', fi, xi)
        weighted_xu = xu * wi[:, np.newaxis, np.newaxis]
        for j in range(h):
            xx[j, j] += np.einsum('bi,bo->io', weighted_xu[:, j, :], weighted_xu[:, j, :])

    xxb = xx * np.outer(b.asnumpy(), b.asnumpy()).reshape((h, h, 1, 1))
    xxr = xxb.transpose(0, 2, 1, 3).reshape(dh, dh)
    xxr1 = xxr + np.eye(dh) * la

    try:
        det = np.linalg.det(xxr1)
        ld = np.log(det)
    except np.linalg.LinAlgError:
        ld = -np.inf

    if ld > -10000:
        A_np = np.linalg.solve(xxr1, buxy.asnumpy())
    else:
        u, s, vt = np.linalg.svd(xxr1)
        s = np.clip(s, 0.0001, None)
        xi = np.dot(u, np.dot(np.diag(s), vt))
        A_np = np.dot(xi, buxy.asnumpy())

    A = ms.Tensor(A_np.reshape(h, d).T, dtype=ms.float32)
    return A

def fitOLS(x, y, la=0.0001):
    """
    使用最小二乘法拟合模型（无权重）。

    参数：
        x (Tensor): 输入特征。
        y (Tensor): 目标值。
        la (float, 可选): 正则化参数。默认值为 0.0001。

    返回：
        Tuple[Tensor, Tensor]: 拟合的系数和截距。
    """
    ones = ops.Ones()((x.shape[0], 1), x.dtype)
    x1 = ops.concat((ones, x), axis=1)
    xx = ops.matmul(x1.transpose(), x1)
    xy = ops.matmul(x1.transpose(), y)
    eye = ops.eye(xx.shape[0], xx.shape[1], dtype=xx.dtype)
    A_matrix = xx + eye * la
    b_sol = ops.matrix_solve(A_matrix, xy)

    if len(b_sol.shape) == 1:
        return b_sol[1:], b_sol[0]
    else:
        return b_sol[1:, :], b_sol[0, :]

def fitA(F, x, y, b, b0, la=0.0001):
    """
    拟合矩阵 A，以最小化损失 ||U*(xA)b + b0 - y||^2。

    参数:
        F (Tensor): 特征矩阵。
        x (Tensor): 输入特征。
        y (Tensor): 目标值。
        b (Tensor): 权重向量。
        b0 (Tensor): 偏置项。
        la (float, 可选): 正则化参数。默认值为 0.0001。

    返回:
        Tuple[Tensor, float]: 拟合的矩阵 A 和对数行列式。
    """
    x_np = x.asnumpy()
    F_np = F.asnumpy()
    y_np = y.asnumpy()

    def data_generator():
        for i in range(x_np.shape[0]):
            sample_x = x_np[i].astype(np.float32)
            sample_F = F_np[i].astype(np.float32)
            sample_y = y_np[i].astype(np.float32)
            assert sample_x.dtype == np.float32, f"Sample x dtype mismatch: {sample_x.dtype}"
            assert sample_F.dtype == np.float32, f"Sample F dtype mismatch: {sample_F.dtype}"
            assert sample_y.dtype == np.float32, f"Sample y dtype mismatch: {sample_y.dtype}"
            yield (sample_x, sample_F, sample_y)

    dataset = GeneratorDataset(source=data_generator, column_names=["x", "F", "y"])
    data_loader = dataset.batch(256, drop_remainder=False).shuffle(buffer_size=10000)

    yb0 = y - b0

    if len(b.shape) == 1:
        ybb = (yb0.reshape(-1, 1)) * (b.reshape(1, -1))
        ybbf = ybb * F
        buxy = ops.matmul(ybbf.T, x)
        buxy = buxy.reshape(-1, 1)
    else:
        ybb = ops.matmul(yb0, b.T)
        bxy = ybb[:, :, np.newaxis] * x[:, np.newaxis, :]
        buxy = np.zeros((x.shape[1] * F.shape[1], 1), dtype=np.float32)
        A_matrix = np.zeros((x.shape[1] * F.shape[1], x.shape[1] * F.shape[1]), dtype=np.float32)

        for batch in data_loader.create_dict_iterator():
            xi = batch["x"].asnumpy()
            fi = batch["F"].asnumpy()
            yi = batch["y"].asnumpy()
            Fx = fi[:, :, np.newaxis] * xi[:, np.newaxis, :]
            Fx_flat = Fx.reshape(xi.shape[0], -1).astype(np.float32)
            A_matrix += np.matmul(Fx_flat.T, Fx_flat).astype(np.float32)
            Fy = fi[:, :, np.newaxis] * yi[:, np.newaxis, :]
            Fy_x = Fy * xi[:, np.newaxis, :]
            Fy_x_flat = Fy_x.reshape(xi.shape[0], -1).astype(np.float32)
            sum_Fy_x_flat = np.sum(Fy_x_flat, axis=0).reshape(-1, 1).astype(np.float32)
            buxy += sum_Fy_x_flat

    A_matrix += np.eye(A_matrix.shape[0], dtype=np.float32) * la

    try:
        det = np.linalg.det(A_matrix)
        ld = np.log(det)
    except np.linalg.LinAlgError:
        ld = -np.inf

    if ld > -10000:
        A_np_flat = np.linalg.solve(A_matrix, buxy)
    else:
        u, s, vt = np.linalg.svd(A_matrix)
        s = np.clip(s, 0.0001, None)
        A_matrix_inv = np.dot(u, np.dot(np.diag(s), vt))
        A_np_flat = np.dot(A_matrix_inv, buxy)

    A_np = A_np_flat.reshape(F.shape[1], x.shape[1]).T
    A = ms.Tensor(A_np, dtype=ms.float32)
    return A, ld
if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    batch_size = 100
    d = 10
    h = 5
    np.random.seed(0)
    x_np = np.random.randn(batch_size, d).astype(np.float32)
    F_np = np.random.randn(batch_size, h).astype(np.float32)
    y_np = np.random.randn(batch_size, 1).astype(np.float32)
    w_np = np.ones((batch_size,), dtype=np.float32)
    b_np = np.random.randn(h, 1).astype(np.float32)
    b0_np = np.random.randn(1).astype(np.float32)
    A_np = np.random.randn(d, h).astype(np.float32)
    al = 0.1

    x = ms.Tensor(x_np)
    F = ms.Tensor(F_np)
    y = ms.Tensor(y_np)
    w = ms.Tensor(w_np)
    b = ms.Tensor(b_np)
    b0 = ms.Tensor(b0_np)
    A = ms.Tensor(A_np)

    # 测试 Fmat
    u = Fmat(A, x)
    print("Fmat 输出形状:", u.shape)

    # 测试 anet1
    py1 = anet1(A, b, b0, u, x)
    print("anet1 输出形状:", py1.shape)

    # 测试 anet
    py = anet(A, b, b0, x, al)
    print("anet 输出形状:", py.shape)

    # 测试 fitOLS（无权重）
    b_fit, b0_fit = fitOLS(x, y, la=0.0001)
    print("fitOLS 系数形状:", b_fit.shape)
    print("fitOLS 截距形状:", b0_fit.shape)

    # 测试 fitAw（计算量大，可能耗时较长）
    A_fit = fitAw(F, x, y, w, b, b0, la=0.0001)
    print("fitAw 拟合的 A 形状:", A_fit.shape)

    # 测试 fitA（计算量大，可能耗时较长）
    A_fitted, log_det = fitA(F, x, y, b, b0, la=0.0001)
    print("fitA 拟合的 A 形状:", A_fitted.shape)
