import numpy as np
from tqdm import tqdm
import math
import mindspore
import mindspore.ops as ops
from mindspore import Tensor, COOTensor


def tri_to_adj(triple, n):
    A = COOTensor(triple[:, :2].int(), triple[:, 2], shape=(n, n)).to_dense()
    A = A + A.T - mindspore.numpy.diag(mindspore.numpy.diag(A))
    return A


def fn_atk_loss(edges, triple_copy):
    triple_torch = ops.cat((triple_copy[:, :2], edges), 1)
    adj = tri_to_adj(triple_torch, len(triple_copy))
    adj_unsign = ops.abs(adj)
    atk_loss = (0.5 * (1 + (ops.trace(ops.matmul(ops.matmul(adj, adj), adj)) / ops.trace(
        ops.matmul(ops.matmul(adj_unsign, adj_unsign), adj_unsign)))))
    return atk_loss


def get_meta_grad(triple_copy):
    edges = Tensor(triple_copy[:, 2:])
    grad_fn = ops.value_and_grad(fn_atk_loss, grad_position=0)
    _, meta_grad = grad_fn(edges, triple_copy)
    return np.concatenate((triple_copy[:, :2], meta_grad), 1)


def gradmax(triple, ptb_rate):
    budget = int(ptb_rate * len(triple))
    triple_copy = mindspore.Tensor(triple)
    flag = True
    perturb = []
    with tqdm(total=budget) as pbar:
        for i in range(math.ceil(budget / 10)):
            pbar.update(10)
            if flag:
                flag = False
            else:
                triple_copy = mindspore.Tensor(triple_copy)
            meta_grad = get_meta_grad(triple_copy)
            v_grad = np.zeros((len(meta_grad), 3))
            for j in range(len(meta_grad)):
                v_grad[j, 0] = meta_grad[j, 0]
                v_grad[j, 1] = meta_grad[j, 1]
                if triple_copy[j, 2] == -1 and meta_grad[j, 2] < 0:
                    v_grad[j, 2] = meta_grad[j, 2]
                elif triple_copy[j, 2] == 1 and meta_grad[j, 2] > 0:
                    v_grad[j, 2] = meta_grad[j, 2]
                else:
                    continue

            v_grad = v_grad[np.abs(v_grad[:, 2]).argsort()]
            K = -1
            triple_copy = triple_copy.asnumpy()
            for k in range(20):
                while v_grad[K][:2].astype('int').tolist() in perturb:
                    K -= 1
                    if abs(K) > len(v_grad):
                        break
                target_grad = v_grad[int(K)]
                target_index = np.where(np.all((triple[:, :2] == target_grad[:2]), axis=1))[0][0]
                triple_copy[target_index, 2] -= 2 * np.sign(target_grad[2])
                perturb.append([int(target_grad[0]), int(target_grad[1])])
                K -= 1
                if abs(K) > len(v_grad):
                    break

    print("output:")
    print(triple_copy)


if __name__ == '__main__':
    mask_ratio = 0.9
    triple = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, -1.0], [2.0, 1.0, 1.0]])
    print(triple)
    gradmax(triple, mask_ratio)
