"""
Model File
"""
import math

from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import HeUniform, Uniform


class CSDNTem(nn.Cell):
    """
    Basc Block
    """

    def __init__(self, in_ch, out_ch):
        super(CSDNTem, self).__init__()

        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=1,
            group=in_ch,
            has_bias=True,
            weight_init=HeUniform(math.sqrt(5)),
            bias_init=Uniform(1 / (9 * in_ch))
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            group=1,
            has_bias=True,
            weight_init=HeUniform(math.sqrt(5)),
            bias_init=Uniform(1 / in_ch)
        )

    def construct(self, x):
        """
        Basic block inference
        """
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class ZeroDCEpp(nn.Cell):
    """
    Zero-DCE++ Model
    """

    def __init__(self, scale_factor):
        super(ZeroDCEpp, self).__init__()
        self.relu = nn.ReLU()
        self.scale_factor = scale_factor
        number_f = 32

        self.cat = ops.Concat(axis=1)

        self.e_conv1 = CSDNTem(3, number_f)
        self.e_conv2 = CSDNTem(number_f, number_f)
        self.e_conv3 = CSDNTem(number_f, number_f)
        self.e_conv4 = CSDNTem(number_f, number_f)
        self.e_conv5 = CSDNTem(number_f * 2, number_f)
        self.e_conv6 = CSDNTem(number_f * 2, number_f)
        self.e_conv7 = CSDNTem(number_f * 2, 3)

    def resize(self, input_, scale_factor):
        """
        resize the input
        """
        h, w = input_.shape[-2::]
        output_size = h * scale_factor, w * scale_factor
        resize_bilinear = ops.ResizeBilinear(output_size)
        return resize_bilinear(input_)

    def enhance(self, x, x_r):
        """
        enhance with the x_r
        """
        x = x + x_r * (ops.pows(x, 2) - x)
        x = x + x_r * (ops.pows(x, 2) - x)
        x = x + x_r * (ops.pows(x, 2) - x)
        enhance_image_1 = x + x_r * (ops.pows(x, 2) - x)
        x = enhance_image_1 + x_r * \
            (ops.pows(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (ops.pows(x, 2) - x)
        x = x + x_r * (ops.pows(x, 2) - x)
        enhance_image = x + x_r * (ops.pows(x, 2) - x)

        return enhance_image

    def construct(self, x):
        """
        inference
        """
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = self.resize(x, 1 / self.scale_factor)

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(self.cat([x3, x4])))
        x6 = self.relu(self.e_conv6(self.cat([x2, x5])))
        x_r = ops.tanh(self.e_conv7(self.cat([x1, x6])))
        if self.scale_factor != 1:
            x_r = self.resize(x_r, self.scale_factor)
        enhance_image = self.enhance(x, x_r)
        return enhance_image, x_r
