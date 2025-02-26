import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import math
import time

EPSILON = 1e-7

# NMF by multiplicative updates
def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=1e-4, verbose=False):
    if verbose:
        start_time = time.time()

    scale = math.sqrt(V.mean().asnumpy() / k)

    if random_seed is not None:
        ms.set_seed(random_seed)

    if W is None:
        W = ops.randn((V.shape[0], k)) * scale
        W = ops.abs(W)  # 确保非负

    update_H = True
    if H is None:
        H = ops.randn((k, V.shape[1])) * scale
        H = ops.abs(H)  # 确保非负
    else:
        update_H = False

    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init

    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_H=update_H, VH=VH, HH=HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error
    if verbose:
        print(f'Exited after {n_iter + 1} iterations. Total time: {time.time() - start_time} seconds')

    return W, H

def multiplicative_update_step(V, W, H, update_H=True, VH=None, HH=None):
    # 更新 W
    if VH is None:
        assert HH is None
        Ht = ops.transpose(H, (1, 0))
        VH = ops.matmul(V, Ht)
        HH = ops.matmul(H, Ht)

    WHH = ops.matmul(W, HH)
    WHH = ops.where(WHH == 0, Tensor(EPSILON, ms.float32), WHH)  # 避免除以零
    W *= VH / WHH

    if update_H:
        # 更新 H（在更新 W 之后）
        Wt = ops.transpose(W, (1, 0))
        WV = ops.matmul(Wt, V)
        WWH = ops.matmul(ops.matmul(Wt, W), H)
        WWH = ops.where(WWH == 0, Tensor(EPSILON, ms.float32), WWH)  # 避免除以零
        H *= WV / WWH
        VH, HH = None, None

    return W, H, VH, HH

# NMF 目标函数（Frobenius 范数）
def approximation_error(V, W, H, square_root=True):
    diff = V - ops.matmul(W, H)
    norm = ops.norm(diff)
    return norm if not square_root else ops.sqrt(norm)
