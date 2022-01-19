"""
The loss module for Zero-DCE.
"""

from mindspore import nn, ops
from mindspore import Tensor
from mindspore import dtype as mstype


class ColorLoss(nn.Cell):
    """
    The Gray world color loss.
    """

    def __init__(self):
        super().__init__()

        self.mean = ops.ReduceMean(keep_dims=True)
        self.split = ops.Split(axis=1, output_num=3)

    def construct(self, x):
        mean_rgb = self.mean(x, [2, 3])
        m_r, m_g, m_b = self.split(mean_rgb)
        d_rg = ops.pows(m_r - m_g, 2)
        d_rb = ops.pows(m_r - m_b, 2)
        d_gb = ops.pows(m_b - m_g, 2)
        k = ops.pows(ops.pows(d_rg, 2) + ops.pows(d_rb, 2) + ops.pows(d_gb, 2), 0.5)

        return k


class SpatialLoss(nn.Cell):
    """
    Spatial Loss
    """
    def __init__(self):
        super().__init__()

        self.left = ops.Conv2D(1, 3, pad_mode="pad", pad=1)
        self.right = ops.Conv2D(1, 3, pad_mode="pad", pad=1)
        self.up = ops.Conv2D(1, 3, pad_mode="pad", pad=1)
        self.down = ops.Conv2D(1, 3, pad_mode="pad", pad=1)

        self.left_weight = Tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=mstype.float32)
        self.right_weight = Tensor([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]]], dtype=mstype.float32)
        self.up_weight = Tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=mstype.float32)
        self.down_weight = Tensor([[[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]], dtype=mstype.float32)

        self.pool = nn.AvgPool2d(4)
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, org, enhance):
        """
        calculate the spatial loss
        """
        org_mean = self.mean(org, 1)
        enhance_mean = self.mean(enhance, 1)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_letf = self.left(org_pool, self.left_weight)
        d_org_right = self.right(org_pool, self.right_weight)
        d_org_up = self.up(org_pool, self.up_weight)
        d_org_down = self.down(org_pool, self.down_weight)

        d_enhance_letf = self.left(enhance_pool, self.left_weight)
        d_enhance_right = self.right(enhance_pool, self.right_weight)
        d_enhance_up = self.up(enhance_pool, self.up_weight)
        d_enhance_down = self.down(enhance_pool, self.down_weight)

        d_left = ops.pows(d_org_letf - d_enhance_letf, 2)
        d_right = ops.pows(d_org_right - d_enhance_right, 2)
        d_up = ops.pows(d_org_up - d_enhance_up, 2)
        d_down = ops.pows(d_org_down - d_enhance_down, 2)

        return d_left + d_right + d_up + d_down


class ExposureLoss(nn.Cell):
    """
    The exposure loss
    """
    def __init__(self, patch_size, mean_val):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        x = self.mean(x, 1)
        mean = self.pool(x)

        return ops.reduce_mean(ops.pows(mean - self.mean_val, 2))


class TotalVariationLoss(nn.Cell):
    """
    Total variation loss
    """
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def construct(self, x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = (x.shape[2] - 1) * x.shape[3]
        count_w = x.shape[2] * (x.shape[3] - 1)
        h_tv = ops.pows((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = ops.pows((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Loss(nn.Cell):
    """
    The total loss of the Zero-DCE
    """
    def __init__(self, patch_size, mean_val):
        super().__init__()

        self.color_loss = ColorLoss()
        self.spatial_loss = SpatialLoss()
        self.exposure_loss = ExposureLoss(patch_size, mean_val)
        self.total_variation_loss = TotalVariationLoss()

    def construct(self, x, y):
        """
        Construct the total loss for zero dce
        """
        enhanced_image, medium = x

        loss_tv = 1600 * self.total_variation_loss(medium)

        loss_spa = ops.reduce_mean(self.spatial_loss(enhanced_image, y))

        loss_col = 5 * ops.reduce_mean(self.color_loss(enhanced_image))

        loss_exp = 10 * ops.reduce_mean(self.exposure_loss(enhanced_image))

        return loss_tv + loss_spa + loss_col + loss_exp
