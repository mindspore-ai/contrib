import mindspore.nn as nn
import mindspore.ops as ops


# Due to the difference in implementation method, there may be very slight differences in result values.
class SoftPooling1D(nn.Cell):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(SoftPooling1D, self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size, strides, "pad", padding, ceil_mode, count_include_pad)

    def construct(self, x):
        x_exp = ops.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class SoftPooling2D(nn.Cell):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, strides, "pad", padding, ceil_mode, count_include_pad,
                                    divisor_override)

    def construct(self, x):
        x_exp = ops.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class SoftPooling3D(nn.Cell):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SoftPooling3D, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size, strides, "pad", padding, ceil_mode, count_include_pad,
                                    divisor_override)

    def construct(self, x):
        x_exp = ops.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


def test():
    # print(SoftPooling1D(2, 2)(ops.ones((1, 1, 32))))
    # print(SoftPooling2D(2, 2)(ops.ones((1, 1, 32, 32))))
    # print(SoftPooling3D(2, 2)(ops.ones((1, 1, 32, 32, 32))))
    # 生成随机数张量
    random_1d = ops.StandardNormal()((1, 1, 32))
    random_2d = ops.StandardNormal()((1, 1, 32, 32))
    random_3d = ops.StandardNormal()((1, 1, 32, 32, 32))
    
    print(SoftPooling1D(2, 2)(random_1d))
    print(SoftPooling2D(2, 2)(random_2d))
    print(SoftPooling3D(2, 2)(random_3d))


if __name__ == '__main__':
    test()
