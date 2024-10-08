import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Parameter, Tensor

class LocalContextNorm(nn.Cell):
    def __init__(self, num_features, channels_per_group=2, window_size=(227, 227), eps=1e-5):
        super(LocalContextNorm, self).__init__()
        # 使用 numpy 创建全零和全一阵列并转换为 Tensor
        self.weight = Parameter(Tensor(np.ones((1, num_features, 1, 1), dtype=np.float32)), name='weight')
        self.bias = Parameter(Tensor(np.zeros((1, num_features, 1, 1), dtype=np.float32)), name='bias')
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size

    def construct(self, x):
        N, C, H, W = x.shape
        G = C // self.channels_per_group
        assert C % self.channels_per_group == 0

        if self.window_size[0] < H and self.window_size[1] < W:
            # Build integral image
            x_squared = x * x
            integral_img = x.cumsum(axis=2).cumsum(axis=3)
            integral_img_sq = x_squared.cumsum(axis=2).cumsum(axis=3)

            # Dilation
            d = (1, self.window_size[0], self.window_size[1])
            integral_img = integral_img.unsqueeze(1)
            integral_img_sq = integral_img_sq.unsqueeze(1)
            kernel = Tensor([[[[[1., -1.], [-1., 1.]]]]], dtype=ms.float32)
            c_kernel = Tensor(1, 1, self.channels_per_group, 1, 1, dtype=ms.float32)

            # Dilated conv
            sums = ops.Conv3D()(integral_img, kernel, stride=(1, 1, 1), dilation=d)
            sums = ops.Conv3D()(sums, c_kernel, stride=(self.channels_per_group, 1, 1))
            squares = ops.Conv3D()(integral_img_sq, kernel, stride=(1, 1, 1), dilation=d)
            squares = ops.Conv3D()(squares, c_kernel, stride=(self.channels_per_group, 1, 1))

            n = self.window_size[0] * self.window_size[1] * self.channels_per_group
            means = sums / n
            var = (squares - sums * sums / n) / n

            _, _, h, w = means.shape
            pad2d = (int((W - w) / 2), int((W - w) / 2), int((H - h) / 2), int((H - h) / 2))
            padded_means = ops.Pad(pad2d, mode='replicate')(means)
            padded_vars = ops.Pad(pad2d, mode='replicate')(var) + self.eps

            for i in range(G):
                x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] = (
                    x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] -
                    padded_means[:, i, :, :].unsqueeze(1)) / (padded_vars[:, i, :, :].unsqueeze(1)).sqrt()

            del integral_img
            del integral_img_sq
        else:
            x = x.view(N, G, -1)
            mean = x.mean(axis=-1)  # 计算平均值
            mean = mean.unsqueeze(-1)  # 添加维度
            var = x.var(axis=-1)  # 计算方差
            var = var.unsqueeze(-1)  # 添加维度
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)

        return x * self.weight + self.bias

# 示例使用
if __name__ == '__main__':
    # 使用 numpy 创建一个全零张量
    x_np = np.zeros((1, 16, 227, 227), dtype=np.float32)

    # 将 numpy 数组转换为 mindspore.Tensor
    x = Tensor(x_np)

    # 实例化 LocalContextNorm 层
    lcn = LocalContextNorm(num_features=16, channels_per_group=2, window_size=(227, 227), eps=1e-5)

    # 前向传播
    y = lcn(x)

    print(y.shape)  # 输出：(1, 16, 227, 227)
