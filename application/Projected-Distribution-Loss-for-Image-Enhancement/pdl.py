from mindspore import ops

EPS = 1e-8


def projectedDistributionLoss(x, y, num_projections=1000):
    def rand_projections(dim, num_projections=1000):
        projections = ops.randn((dim, num_projections))
        projections = projections / ops.sqrt(
            ops.sum(projections ** 2, dim=0, keepdim=True))  # columns are unit length normalized
        return projections

    x = x.reshape(x.shape[0], x.shape[1], -1)  # B,N,M
    y = y.reshape(y.shape[0], y.shape[1], -1)
    W = rand_projections(x.shape[-1], num_projections=num_projections)
    e_x = ops.matmul(x, W)  # multiplication via broad-casting
    e_y = ops.matmul(y, W)
    loss = 0
    for ii in range(e_x.shape[2]):
        loss = loss + ops.l1_loss(ops.sort(e_x[:, :, ii], axis=1)[0],
                                  ops.sort(e_y[:, :, ii], axis=1)[0])  # if this gives issues; try Huber loss later
    return loss


if __name__ == '__main__':
    x = ops.randn((2, 3, 4, 5))
    y = ops.randn((2, 3, 4, 5))
    r = projectedDistributionLoss(x, y)
    print(r)
