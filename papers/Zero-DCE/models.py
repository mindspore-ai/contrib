"""
Model File
"""

from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Normal

class ZeroDCE(nn.Cell):
    """
    Main Zero DCE Model
    """
    def __init__(self, *, sigma=0.02, mean=0.0):
        super().__init__()

        self.relu = nn.ReLU()

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, pad_mode='pad', padding=1, has_bias=True,
                                 weight_init=Normal(sigma, mean))
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, pad_mode='pad', padding=1, has_bias=True,
                                 weight_init=Normal(sigma, mean))
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, pad_mode='pad', padding=1, has_bias=True,
                                 weight_init=Normal(sigma, mean))
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, pad_mode='pad', padding=1, has_bias=True,
                                 weight_init=Normal(sigma, mean))
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, pad_mode='pad', padding=1,
                                 has_bias=True, weight_init=Normal(sigma, mean))
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, pad_mode='pad', padding=1,
                                 has_bias=True, weight_init=Normal(sigma, mean))
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, pad_mode='pad', padding=1, has_bias=True,
                                 weight_init=Normal(sigma, mean))

        self.split = ops.Split(axis=1, output_num=8)
        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        """
        ZeroDCE inference
        """
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(self.cat([x3, x4])))
        x6 = self.relu(self.e_conv6(self.cat([x2, x5])))

        x_r = ops.tanh(self.e_conv7(self.cat([x1, x6])))
        r1, r2, r3, r4, r5, r6, r7, r8 = self.split(x_r)

        x = x + r1 * (ops.pows(x, 2) - x)
        x = x + r2 * (ops.pows(x, 2) - x)
        x = x + r3 * (ops.pows(x, 2) - x)
        enhance_image_1 = x + r4 * (ops.pows(x, 2) - x)
        x = enhance_image_1 + r5 * (ops.pows(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (ops.pows(x, 2) - x)
        x = x + r7 * (ops.pows(x, 2) - x)
        enhance_image = x + r8 * (ops.pows(x, 2) - x)
        r = self.cat([r1, r2, r3, r4, r5, r6, r7, r8])
        return enhance_image, r
