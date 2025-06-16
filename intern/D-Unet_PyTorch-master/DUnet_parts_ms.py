import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class SE_block(nn.Cell):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Dense(in_channels, in_channels // ratio, has_bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(in_channels // ratio, in_channels, has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()

    def construct(self, x):
        b, c, _, _ = x.shape
        # Global Average Pooling
        y = self.global_avg_pool(x)  # [b, c, 1, 1]
        y = self.reshape(y, (b, c))  # [b, c]
        
        # FC layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Reshape and multiply
        y = self.reshape(y, (b, c, 1, 1))
        return x * y

class BN_block2d(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ])

    def construct(self, x):
        return self.block(x)

class BN_block3d(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.SequentialCell([
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ])

    def construct(self, x):
        return self.block(x)

class D_SE_Add(nn.Cell):
    def __init__(self, in_channels_2d, out_channels, mid_channels_3d):
        super().__init__()
        self.se_2d = SE_block(in_channels_2d)
        self.conv3d_1x1 = nn.Conv3d(mid_channels_3d, mid_channels_3d, kernel_size=1, has_bias=False)
        
        self.conv2d = nn.Conv2d(mid_channels_3d, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.relu = nn.ReLU()
        self.se_3d_to_2d = SE_block(out_channels)
        
        # 操作符
        self.reduce_mean_op = ops.ReduceMean(keep_dims=False)
        self.squeeze_op = ops.Squeeze(2)

    def construct(self, x_3d, x_2d):
        x_2d = self.se_2d(x_2d)
        x_3d = self.conv3d_1x1(x_3d)  # [N, C, D, H, W]
        
        # 3D转2D：在深度维度取平均或直接squeeze
        if x_3d.shape[2] == 1:
            x_3d = self.squeeze_op(x_3d)  # [N, C, 1, H, W] -> [N, C, H, W]
        else:
            x_3d = self.reduce_mean_op(x_3d, 2)  # [N, C, D, H, W] -> [N, C, H, W]
        
        x_3d = self.conv2d(x_3d)
        x_3d = self.relu(x_3d)
        x_3d = self.se_3d_to_2d(x_3d)
        
        return x_2d + x_3d

class UpBlock(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.relu = nn.ReLU()

    def construct(self, x):
        _, _, h, w = x.shape
        target_h, target_w = h * 2, w * 2

        x = ops.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.relu(x)
        return x