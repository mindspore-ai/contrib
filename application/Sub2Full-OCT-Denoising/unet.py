from mindspore import nn, ops, Tensor
import numpy as np


class UNet(nn.Cell):
    """Use the same U-Net architecture as in Noise2Noise (Lehtinen et al., 2018)."""

    def __init__(self, in_channels=1, out_channels=1, feature_maps=48):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.SequentialCell(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.feature_maps, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.feature_maps, out_channels=self.feature_maps, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.SequentialCell(
            nn.Conv2d(in_channels=self.feature_maps, out_channels=self.feature_maps, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.SequentialCell(
            nn.Conv2d(in_channels=self.feature_maps, out_channels=self.feature_maps, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            # mindspore 目前仅支持四维tensor的area插值
            nn.Upsample(scale_factor=2.0, mode='area'))  # Upsample

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.SequentialCell(
            nn.Conv2d(in_channels=self.feature_maps*2, out_channels=self.feature_maps*2, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.feature_maps*2, out_channels=self.feature_maps*2, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2.0, mode='area'))  # Upsample

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.SequentialCell(
            nn.Conv2d(in_channels=self.feature_maps*3, out_channels=self.feature_maps*2, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=self.feature_maps*2, out_channels=self.feature_maps*2, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2.0, mode='area'))  # Upsample

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.SequentialCell(
            nn.Conv2d(in_channels=self.feature_maps*2 + self.in_channels, out_channels=64, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.Identity())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.bias.set_data(Tensor(np.zeros_like(m.bias.asnumpy())))

    def construct(self, x):
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = ops.operations.Concat(1)([upsample5, pool4])
        upsample4 = self._block4(concat5)
        concat4 = ops.operations.Concat(1)([upsample4, pool3])
        upsample3 = self._block5(concat4)
        concat3 = ops.operations.Concat(1)([upsample3, pool2])
        upsample2 = self._block5(concat3)
        concat2 = ops.operations.Concat(1)([upsample2, pool1])
        upsample1 = self._block5(concat2)
        concat1 = ops.operations.Concat(1)([upsample1, x])
        out = self._block6(concat1)

        return out


if __name__ == '__main__':
    model = UNet()
    x = Tensor(np.random.rand(16, 1, 384, 384).astype(np.float32))
    output = model(x)
    print(f"Output shape: {output.shape}")
