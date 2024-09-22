import mindspore as ms
from mindspore import ops,Tensor


def map_xi(x):
    x_x = x[..., :-1]
    x_y = ops.exp(x[..., -1] / x.shape[-1])
    return x_x * x_y.unsqueeze(-1), x_y


def umbral(q, k, r=1, gamma=1):
    q_x, q_y = map_xi(q)
    k_x, k_y = map_xi(k)
    q_y = q_y.unsqueeze(2)
    k_y = k_y.unsqueeze(1)
    out = ops.maximum(ops.maximum(q_y, k_y),
                        (ops.cdist(q_x, k_x) / ops.sinh(Tensor(r,ms.float16)) +
                         ops.add(q_y, k_y)) / 2)
    return -gamma * out

if __name__ == '__main__':
    # [b, n, d]
    q = ops.randn(100, 10, 2)
    k = ops.randn(100, 10, 2)

    # [100, 10, 10]
    print(umbral(q, k).shape)
