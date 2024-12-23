import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class CausalConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CausalConvBlock, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                pad_mode='valid'
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout(p=0.5),  # 修改为 p=0.5
            nn.LeakyReLU(alpha=0.01)  # MindSpore 不支持 inplace
        ])

    def construct(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F_out, T_out]
        """
        return self.conv(x)


class CausalTransConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CausalTransConvBlock, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2dTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                pad_mode='valid'
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(alpha=0.01)
        ])

    def construct(self, x):
        """
        2D Causal transposed convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F_out, T_out]
        """
        return self.conv(x)


class CUNET(nn.Cell):
    """
    Input: [B, C, F, T]
    Output: [B, C, T, F]
    """

    def __init__(self):
        super(CUNET, self).__init__()
        self.conv_block_1 = CausalConvBlock(8, 32, (6, 2), (2, 1))
        self.conv_block_2 = CausalConvBlock(32, 32, (7, 2), (2, 1))
        self.conv_block_3 = CausalConvBlock(32, 64, (7, 2), (2, 1))
        self.conv_block_4 = CausalConvBlock(64, 64, (6, 2), (2, 1))
        self.conv_block_5 = CausalConvBlock(64, 96, (6, 2), (2, 1))
        self.conv_block_6 = CausalConvBlock(96, 96, (6, 2), (2, 1))
        self.conv_block_7 = CausalConvBlock(96, 128, (2, 2), (2, 1))
        self.conv_block_8 = CausalConvBlock(128, 256, (2, 2), (1, 1))

        self.tran_conv_block_1 = CausalTransConvBlock(256, 256, (2, 2), (1, 1))
        self.tran_conv_block_2 = CausalTransConvBlock(256 + 128, 128, (2, 2), (2, 1))
        self.tran_conv_block_3 = CausalTransConvBlock(128 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_4 = CausalTransConvBlock(96 + 96, 96, (6, 2), (2, 1))
        self.tran_conv_block_5 = CausalTransConvBlock(96 + 64, 64, (6, 2), (2, 1))
        self.tran_conv_block_6 = CausalTransConvBlock(64 + 64, 64, (7, 2), (2, 1))
        self.tran_conv_block_7 = CausalTransConvBlock(64 + 32, 32, (7, 2), (2, 1))
        self.tran_conv_block_8 = CausalTransConvBlock(32 + 32, 32, (6, 2), (2, 1))
        self.last_conv_block = nn.SequentialCell([
            nn.Conv2d(
                in_channels=32,
                out_channels=8,
                kernel_size=1,
                stride=1,
                padding=0,
                pad_mode='valid'
            ),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(alpha=0.01)
        ])

        self.dense = nn.SequentialCell([
            nn.Dense(514, 514),
            nn.Sigmoid()
        ])

        self.concat = ops.Concat(axis=1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.slice = ops.Slice()
        self.shape = ops.Shape()

    def construct(self, x):
        e1 = self.conv_block_1(x)
        e2 = self.conv_block_2(e1)
        e3 = self.conv_block_3(e2)
        e4 = self.conv_block_4(e3)
        e5 = self.conv_block_5(e4)
        e6 = self.conv_block_6(e5)
        e7 = self.conv_block_7(e6)
        e8 = self.conv_block_8(e7)

        d = self.tran_conv_block_1(e8)
        d = self.concat((d, e7))
        d = self.tran_conv_block_2(d)

        d = self.concat((d, e6))
        d = self.tran_conv_block_3(d)

        d = self.concat((d, e5))
        d = self.tran_conv_block_4(d)

        d = self.concat((d, e4))
        d = self.tran_conv_block_5(d)

        d = self.concat((d, e3))
        d = self.tran_conv_block_6(d)

        d = self.concat((d, e2))
        d = self.tran_conv_block_7(d)

        d = self.concat((d, e1))
        d = self.tran_conv_block_8(d)

        d = self.last_conv_block(d)
        d = self.transpose(d, (0, 1, 3, 2))  # [B, C, T, F]
        d = d[:, :, :-8, :]  # [B, C, T-8, F]

        B, C, T_new, F = d.shape
        # print(f"After slicing: B={B}, C={C}, T_new={T_new}, F={F}")

        d_reshaped = self.reshape(d, (B * C * T_new, F))
        d_dense = self.dense(d_reshaped)
        d = self.reshape(d_dense, (B, C, T_new, F))
        return d


def l2_norm(s1, s2):
    norm = ops.ReduceSum(keep_dims=True)(s1 * s2, -1)
    return norm


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    # 使用自然对数，转换为 base 10
    snr = 10 * ops.Log()(target_norm / (noise_norm + eps) + eps) / ops.Log()(ms.Tensor(np.e, dtype=ms.float32))
    return ops.ReduceMean()(snr)


def snr_loss(inputs, label):
    return -si_snr(inputs, label)


def MAE_loss(inputs, label):
    diff = ops.Abs()(inputs - label)
    return ops.ReduceSum()(diff) / inputs.shape[-1]


if __name__ == '__main__':
    layer = CUNET()
    layer.set_train(False)
    K = 8
    x_np = np.random.rand(1, 8, 514, 249).astype(np.float32)
    x = ms.Tensor(x_np)
    print("input shape:", x.shape)
    prefix_frames = ms.Tensor(np.zeros((1, 8, 514, K), dtype=np.float32))
    y = ops.Concat(axis=3)((x, prefix_frames))
    print("concatenated input shape:", y.shape)
    output = layer(y)
    print("output shape:", output.shape)
    total_num = 0
    for param in layer.get_parameters():
        total_num += np.prod(param.asnumpy().shape)
    print("Total number of parameters:", total_num)