"""
The loss for UColor
"""

from mindspore import nn, Tensor, dtype
import numpy as np

from vgg import vgg16_pretrain_features

IMAGE_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
IMAGE_STD = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])


class Loss(nn.Cell):
    """
    Loss
    """
    def __init__(self, vgg_ckpt_file):
        super(Loss, self).__init__()

        self.vgg = vgg16_pretrain_features(vgg_ckpt_file)
        self.mse = nn.MSELoss()
        self.image_mean = Tensor(
            IMAGE_MEAN, dtype=dtype.float32).reshape((1, 3, 1, 1))
        self.image_std = Tensor(
            IMAGE_STD, dtype=dtype.float32).reshape((1, 3, 1, 1))

    def vgg_prepross(self, x):
        """
        norm the data
        """
        return (x * 255.0 - self.image_mean) / self.image_std

    def construct(self, x, gt):
        """
        calculate the loss
        """
        perceptual_loss = self.mse(
            self.vgg(self.vgg_prepross(x)), self.vgg(self.vgg_prepross(gt)))
        mse_loss = self.mse(x, gt)

        return 5 * mse_loss + 0.05 * perceptual_loss
