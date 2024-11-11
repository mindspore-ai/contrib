import mindspore
import mindspore.ops as ops
from mindspore import Tensor,context
import numpy as np
import cv2
from time import time


def SingleAnchorDPR(centerx, centery, r, num_regions, h, w, vis=False, cc=-100):
    '''
    :param centerx: [bs]
    :param centery: [bs]
    :param r: [bs, num_regions], require_grad=True
    :param num_regions: int > 3
    :param (h, w): (int, int) 
    :return: act, [bs, h, w]
    '''
    
    assert r.shape[1] == num_regions, 'the len of r should be equal with num_regions.'
    bs = r.shape[0]
    # Get the matrix of coordinate
    coordinate = ops.zeros((bs, h, w, 2))
    coordinate[:, :, :, 0] = (ops.arange(h).tile((w, 1))).T
    coordinate[:, :, :, 1] = ops.arange(w).tile((h, 1))
    # Generate the mask of regions
    relate_cx = coordinate[:, :, :, 0] - centerx.view(bs, 1, 1)
    relate_cy = coordinate[:, :, :, 1] - centery.view(bs, 1, 1)
    mask_region = (ops.acos(relate_cy / (ops.sqrt(relate_cx ** 2 + relate_cy ** 2))) / np.pi * 180)
    dis = ops.sqrt(relate_cx ** 2 + relate_cy ** 2)

    sub_angle_index = (relate_cx < 0)
    mask_region[sub_angle_index] = 2 * 180 - mask_region[sub_angle_index]
    mask_region = mask_region // (360 // num_regions)

    batch_indices = Tensor(np.arange(bs), mindspore.int64)
    mask_region[batch_indices, centerx, centery] = num_regions
    mask_region = mask_region.long()

    # Calculate points
    points =ops.zeros((bs, num_regions, 2)) # [bs, num_regions, 2]
    act = ops.zeros((bs, h, w), mindspore.float32)
    #act = mindspore.Parameter(Tensor((np.zeros((bs, h, w))), dtype=mindspore.float32))
    
    for i in range(num_regions):
        angle = (360 / num_regions * i / 360 * 2 * Tensor([np.pi],dtype = mindspore.float32))
        points[:, i, 0] = centerx * 1.0 + r[:, i] * ops.sin(angle)
        points[:, i, 1] = centery * 1.0 + r[:, i] * ops.cos(angle)

    # Calculate Act
    for i in range(num_regions):
        idx = ops.nonzero(mask_region == i)
        if (num_regions - 1) == i:
            a = points[:, 0]
            b = points[:, i]
        else:
            a = points[:, i]
            b = points[:, i + 1]
        A, B, C = GaussianElimination(a, b)

        bs_idx = idx[:, 0]
        c = idx[:, 1:]
        tx = centerx[bs_idx]
        ty = centery[bs_idx]
        o = ops.stack((tx, ty), axis=1).float()
        A1, B1, C1 = GaussianElimination(o, c.float())

        A0 = A[bs_idx]
        B0 = B[bs_idx]
        C0 = C[bs_idx]

        D = A0 * B1 - A1 * B0
        x = (B0 * C1 - B1 * C0) * 1.0 / D
        y = (A1 * C0 - A0 * C1) * 1.0 / D

        assert ops.isnan(x).long().sum() == 0, 'Calculate act has been found None!'

        before_act = dis[bs_idx, c[:, 0], c[:, 1]] / ops.sqrt(
            (1.0 * o[:, 0] - x) ** 2 + (1.0 * o[:, 1] - y) ** 2)
        act[bs_idx, c[:, 0], c[:, 1]] = ActFunc(before_act, cc=cc)
        
    batch_indices = Tensor(np.arange(bs), mindspore.int64)
    act[batch_indices, centerx, centery] = 1
    return act

def ActFunc(x, cc=-100):
    ans = (ops.tanh(cc * (x - 1)) + 1) / 2
    return ans

def GaussianElimination(a, b):
    # Ax+By+C=0
    first_x, first_y, second_x, second_y = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    A = 1.0 * second_y - 1.0 * first_y
    B = 1.0 * first_x - 1.0 * second_x
    C = 1.0 * second_x * first_y - 1.0 * first_x * second_y
    return A, B, C

def loss_for_mindspore( centerx, centery, r, num_regions, h, w):
    #define backward for mindspore
    act = SingleAnchorDPR(centerx, centery, r, num_regions, h, w, vis=False, cc=-100)
    mask = ops.stack((act, act, act), axis=1)
    return mask


if __name__ == '__main__':

    np.set_printoptions(precision=3)
    bs, c, h, w = 100, 3, 100, 100
    patch_size = int(np.sqrt(h * w * 0.03))
    radius = patch_size // 2
    centerx = Tensor(np.random.randint(radius, h - radius, bs), mindspore.int64)
    centery = Tensor(np.random.randint(radius, w - radius, bs), mindspore.int64)
    num_regions = 36
    tmp = np.ones((bs, num_regions)) * radius
    r = mindspore.Parameter(Tensor(tmp, dtype=mindspore.float32))
    start = time()
    grad_func = mindspore.grad(loss_for_mindspore, grad_position=2)
    r_grad = grad_func(centerx, centery, r, num_regions, h, w)
    print(r_grad)
    print(time() - start)


