import mindspore.nn as nn
from mindspore import ops, Parameter, Tensor


class DyReLU(nn.Cell):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Dense(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.lambdas = Parameter(Tensor([1.]*k + [0.5]*k).float(), name='lambdas', requires_grad=False)
        self.init_v = Parameter(Tensor([1.] + [0.]*(2*k - 1)).float(), name='init_v', requires_grad=False)

    def get_relu_coefs(self, x):
        theta = ops.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = ops.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def construct(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Dense(channels // reduction, 2*k)

    def construct(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.swapaxes(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = ops.max(output, axis=-1)[0].swapaxes(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Dense(channels // reduction, 2*k*channels)

    def construct(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = ops.max(output, axis=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = ops.max(output, axis=-1)[0].permute(2, 3, 0, 1)

        return result


if __name__ == '__main__':
    conv = nn.Conv2d(3, 10, 5)
    relu_a = DyReLUB(10, conv_type='2d')
    relu_b = DyReLUB(10, conv_type='2d')

    x = ops.randn([1, 3, 224, 224])
    x = conv(x)
    output_a = relu_a(x)
    output_b = relu_b(x)

    print(output_a.shape)
    print(output_b.shape)
    print(output_a)
    print(output_b)
