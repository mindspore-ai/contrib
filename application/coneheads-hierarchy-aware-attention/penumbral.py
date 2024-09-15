import mindspore
from mindspore import ops


def map_psi(x, r):
    x_x = x[..., :-1]
    x_y = ops.sigmoid(x[..., -1])
    return x_x * x_y.unsqueeze(-1) * r, x_y * r


def penumbral(q, k, r=1, gamma=1, eps=1e-6):
    q_x, q_y = map_psi(q, r)
    k_x, k_y = map_psi(k, r)
    q_y = q_y.unsqueeze(2)
    k_y = k_y.unsqueeze(1)

    x_q_y = ops.sqrt(r**2 - q_y**2 + eps)
    x_k_y = ops.sqrt(r**2 - k_y**2 + eps)

    pairwise_dist = ops.cdist(q_x, k_x)

    lca_height = ops.maximum(ops.maximum(q_y**2, k_y**2),
                               r**2 - ((x_q_y + x_k_y - pairwise_dist) / 2)**2)

    lca_height_outcone = ((pairwise_dist**2 + k_y**2 - q_y**2) /
                          (2 * pairwise_dist + eps))**2 + q_y**2

    exists_cone = ops.logical_or(pairwise_dist <= x_q_y,
                                   (pairwise_dist - x_q_y)**2 + k_y**2 <= r**2)

    return -gamma * ops.where(exists_cone, lca_height, lca_height_outcone)

if __name__ == '__main__':
    # [b, n, d]
    q = ops.randn(100, 10, 2)
    k = ops.randn(100, 10, 2)

    # [100, 10, 10]
    print(penumbral(q, k).shape)