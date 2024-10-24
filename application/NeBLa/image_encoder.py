import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


ms.set_context(device_target='Ascend', device_id=0)


# DoubleConv block for UNET
class DoubleConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def construct(self, x):
        return self.conv(x)
    
    
class UNET(nn.Cell):
    def __init__(
            self, in_channels=1, out_channels=128, features=[64, 128, 256, 512], # 128, [64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.ups = nn.CellList()
        self.downs = nn.CellList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='pad')

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.Conv2dTranspose(
                    feature*2, feature, kernel_size=2, stride=2, pad_mode='pad'
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, pad_mode='pad')

    def construct(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = ops.interpolate(x, skip_connection.shape[2:])

            concat_skip = ops.cat((skip_connection, x), axis=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    

def main():
    
    in_channels = 3
    out_channels = 128
    features = [64, 128, 256, 512]
    batch_size = 1
    img_size = 256

    x = ops.randn((batch_size, in_channels, img_size, img_size))
    model = UNET(in_channels, out_channels, features)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    
if __name__ == "__main__":
    main()
    