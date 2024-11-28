import mindspore
from mindspore import nn, ops


class ssFPN(nn.Cell):
    def __init__(self, channel_in, channel_out=None, bn=True, act=True):
        super().__init__()
        channel_out = channel_out if channel_out else channel_in
        self.upx2 = nn.Upsample(scale_factor=2., recompute_scale_factor=True)
        self.upx4 = nn.Upsample(scale_factor=4., recompute_scale_factor=True)
        self.conv = nn.Conv3d(channel_in, channel_out,
                              (1, 3, 3), stride=1, pad_mode="pad", padding=1, bias_init=False)
        self.bn = nn.BatchNorm3d(channel_out) if bn else nn.Identity()
        self.act = nn.LeakyReLU(alpha=0.01) if act else nn.Identity()
        self.avg_pool = nn.AvgPool3d((3, 1, 1))

    def construct(self, x):
        p3, p4, p5 = x
        x = ops.cat([p3, self.upx2(p4), self.upx4(p5)], 1).unsqueeze(2)
        x = self.act(self.bn(self.conv(x)))
        return self.avg_pool(x).squeeze(2)


if __name__ == "__main__":
    model = ssFPN(140, 20)
    p3, p4, p5 = ops.randn((1, 20, 80, 80)), ops.randn(
        (1, 40, 40, 40)), ops.randn((1, 80, 20, 20))
    output = model((p3, p4, p5))
    print(model, output.shape)
