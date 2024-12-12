import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
import numpy as np


def raw_depth_attention(x):
    """ x: input features with shape [N, C, H, W] """
    N, C, H, W = x.shape
    k = 7
    adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
    # channel=9
    conv = nn.Conv2d(9, 9, kernel_size=(k, 1), pad_mode='same', has_bias=True)
    softmax = ops.Softmax(axis=-2)

    x_pool = adaptive_pool(x)
    x_transpose = ops.Transpose()(x_pool, (0, 1, 3, 2))
    y = conv(x_transpose)
    y = softmax(y)
    y = ops.Transpose()(y, (0, 1, 3, 2))
    
    return y * C * x


class TSFF(nn.Cell):
    def __init__(self, img_weight=0.02, width=224, length=224, num_classes=2, samples=1000, channels=3, avepool=25):
        super(TSFF, self).__init__()

        self.channel_weight = Parameter(initializer('XavierUniform', [9, 1, 3], mindspore.dtype.float32), name="w1")

        self.num_classes = num_classes
        self.img_weight = img_weight

        self.raw_time_conv = nn.SequentialCell([
            nn.Conv2d(9, 24, kernel_size=(1, 1), group=1, has_bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, 75), group=24, pad_mode='valid', has_bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
        ])

        self.raw_chanel_conv = nn.SequentialCell([
            nn.Conv2d(24, 9, kernel_size=(1, 1), group=1, has_bias=False),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(channels, 1), group=9, pad_mode='valid', has_bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
        ])

        self.raw_norm = nn.SequentialCell([
            nn.AvgPool3d(kernel_size=(1, 1, avepool), stride=(1, 1, avepool)),
            nn.Dropout(p=0.35),
        ])

        raw_eeg = Tensor(mindspore.numpy.ones((1, 1, channels, samples)), mindspore.float32)
        # raw_eeg = ops.einsum('bdcw, hdc->bhcw', raw_eeg, self.channel_weight)
        # 修改为自定义的计算（mindspore的cpu尚未支持einsum）
        channel_weight_exp = ops.ExpandDims()(self.channel_weight, 0)  # (h, d, c) -> (1, h, d, c)
        channel_weight_exp = ops.ExpandDims()(channel_weight_exp, -1) # (1, h, d, c) -> (1, h, d, c, 1)
        raw_eeg_exp = ops.ExpandDims()(raw_eeg, 1)  # (b, d, c, w) -> (b, 1, d, c, w)
        result = raw_eeg_exp * channel_weight_exp  # (b, h, d, c, w)
        result = ops.ReduceSum()(result, 2)
        raw_eeg = result

        out_raw_eeg = self.raw_time_conv(raw_eeg)
        out_raw_eeg = self.raw_chanel_conv(out_raw_eeg)
        out_raw_eeg = self.raw_norm(out_raw_eeg)
        out_raw_eeg_shape = out_raw_eeg.shape
        print('out_raw_eeg_shape: ', out_raw_eeg_shape)
        n_out_raw_eeg = out_raw_eeg_shape[-1] * out_raw_eeg_shape[-2] * out_raw_eeg_shape[-3]

        self.frequency_features = nn.SequentialCell([
            nn.Conv2d(3, 16, kernel_size=(4, 4), stride=1, pad_mode="pad", padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=8),
            nn.Dropout(p=0.75),

            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=1, pad_mode="pad", padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Dropout(p=0.75),

            nn.Conv2d(32, out_raw_eeg_shape[-1], kernel_size=1, has_bias=False),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.Conv2d(out_raw_eeg_shape[-1], out_raw_eeg_shape[-1], kernel_size=4,
                      group=out_raw_eeg_shape[-1], has_bias=False, pad_mode="pad", padding=2),
            nn.BatchNorm2d(out_raw_eeg_shape[-1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Dropout(p=0.75),
        ])

        img_eeg = Tensor(mindspore.numpy.ones((1, 3, width, length)), mindspore.float32)
        out_img = self.frequency_features(img_eeg)
        out_img_shape = out_img.shape
        n_out_img = out_img_shape[-1] * out_img_shape[-2] * out_img_shape[-3]
        print('n_out_img shape: ', out_img_shape)

        self.classifier = nn.SequentialCell([
            nn.Dense(n_out_img, num_classes),
        ])


    def construct(self, x_raw, x_frequency):
        # Features for frequency graph
        x_frequency = self.frequency_features(x_frequency)
        x_frequency = ops.reshape(x_frequency, (x_frequency.shape[0], -1))

        # x_raw = ops.einsum('bdcw, hdc->bhcw', x_raw, self.channel_weight)
        # 修改为自定义的计算（mindspore的cpu尚未支持einsum）
        channel_weight_exp = ops.ExpandDims()(self.channel_weight, 0)  # (h, d, c) -> (1, h, d, c)
        channel_weight_exp = ops.ExpandDims()(channel_weight_exp, -1) # (1, h, d, c) -> (1, h, d, c, 1)
        raw_eeg_exp = ops.ExpandDims()(x_raw, 1)  # (b, d, c, w) -> (b, 1, d, c, w)
        result = raw_eeg_exp * channel_weight_exp  # (b, h, d, c, w)
        result = ops.ReduceSum()(result, 2)
        x_raw = result

        x_raw = self.raw_time_conv(x_raw)
        x_raw = self.raw_chanel_conv(x_raw)
        x_raw = raw_depth_attention(x_raw)
        x_raw = self.raw_norm(x_raw)

        # Flatten and weight features
        x_raw_flatten = ops.reshape(x_raw, (x_raw.shape[0], -1))
        weighted_features = x_raw_flatten * (1 - self.img_weight) + x_frequency * self.img_weight

        x = self.classifier(weighted_features)

        return x, x_raw_flatten, x_frequency
    

def main():
    img_weight = 0.02
    width, length = 224, 224
    num_classes = 2
    samples = 1000
    channels = 3
    avepool = 25

    model = TSFF(img_weight=img_weight, width=width, length=length, num_classes=num_classes,
                 samples=samples, channels=channels, avepool=avepool)

    x_raw = Tensor(np.random.randn(16, 9, channels, samples).astype(np.float32))
    x_frequency = Tensor(np.random.randn(16, 3, width, length).astype(np.float32))

    output, x_raw_flatten, x_frequency = model(x_raw, x_frequency)

    print("Output shape:", output.shape)
    print("x_raw_flatten shape:", x_raw_flatten.shape)
    print("x_frequency shape:", x_frequency.shape)


if __name__ == '__main__':
    main()
