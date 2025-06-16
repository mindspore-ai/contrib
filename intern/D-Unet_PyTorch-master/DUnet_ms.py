import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeUniform, Zero, initializer

from DUnet_parts_ms import *
from loss_ms import *

def weights_init_he(cell):
    for _, child in cell.cells_and_names():
        if isinstance(child, (nn.Conv2d, nn.Conv3d)):
            child.weight.set_data(initializer(HeUniform(), child.weight.shape, child.weight.dtype))
            if hasattr(child, 'bias') and child.bias is not None:
                child.bias.set_data(initializer(Zero(), child.bias.shape, child.bias.dtype))

class DUnet_ms(nn.Cell):
    def __init__(self, in_channels_2d=4, in_channels_3d=1, weights_init=True):
        super().__init__()
        
        self.in_channels_2d = in_channels_2d
        self.in_channels_3d = in_channels_3d
        
        self.maxpool_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        
        # 3d down
        self.bn_3d_1 = BN_block3d(in_channels_3d, 32)
        self.bn_3d_2 = BN_block3d(32, 64)
        self.bn_3d_3 = BN_block3d(64, 128)
        
        # 2d down
        self.bn_2d_1 = BN_block2d(in_channels_2d, 32)
        
        self.bn_2d_2 = BN_block2d(32, 64)
        self.se_add_2 = D_SE_Add(64, 64, 64)  # (2d_channels, out_channels, 3d_channels)
        
        self.bn_2d_3 = BN_block2d(64, 128)
        self.se_add_3 = D_SE_Add(128, 128, 128)
        
        self.bn_2d_4 = BN_block2d(128, 256)
        self.bn_2d_5 = BN_block2d(256, 512)
        
        # up
        self.up_1 = UpBlock(512, 256)
        self.bn_2d_6 = BN_block2d(512, 256)  # 256 + 256 = 512
        
        self.up_2 = UpBlock(256, 128)
        self.bn_2d_7 = BN_block2d(256, 128)  # 128 + 128 = 256
        
        self.up_3 = UpBlock(128, 64)
        self.bn_2d_8 = BN_block2d(128, 64)   # 64 + 64 = 128
        
        self.up_4 = UpBlock(64, 32)
        self.bn_2d_9 = BN_block2d(64, 32)    # 32 + 32 = 64
        
        self.final_conv = nn.SequentialCell([
            nn.Conv2d(32, 1, kernel_size=1, has_bias=False),
            nn.Sigmoid()
        ])
        
        # 拼接操作
        self.concat = ops.Concat(axis=1)
        
        # 操作符定义
        self.reduce_mean_op = ops.ReduceMean(keep_dims=True)
        self.expand_dims_op = ops.ExpandDims()
        self.tile_op = ops.Tile()
        
        # He initialization stated in the original paper
        if weights_init:
            weights_init_he(self)

    def construct(self, x_2d, x_3d=None):   
        # 如果没有提供3D输入，则从2D输入生成
        if x_3d is None:
            # 创建伪3D输入 - 用于演示，实际应用中应该使用真实的3D多模态数据
            x_3d = self.reduce_mean_op(x_2d, 1)  # [batch, 1, H, W] - 修复参数调用
            x_3d = self.expand_dims_op(x_3d, 2)  # [batch, 1, 1, H, W]
            # 为了避免池化后维度消失，我们复制深度层
            x_3d = self.tile_op(x_3d, (1, 1, 8, 1, 1))  # [batch, 1, 8, H, W]
        
        # 3d Stream
        conv3d_1 = self.bn_3d_1(x_3d)       # [batch, 32, 8, H, W]
        pool3d_1 = self.maxpool_3d(conv3d_1)  # [batch, 32, 4, H/2, W/2]
        
        conv3d_2 = self.bn_3d_2(pool3d_1)   # [batch, 64, 4, H/2, W/2]
        pool3d_2 = self.maxpool_3d(conv3d_2)  # [batch, 64, 2, H/4, W/4]
        
        conv3d_3 = self.bn_3d_3(pool3d_2)   # [batch, 128, 2, H/4, W/4]
        
        # 2d Encoding
        conv2d_1 = self.bn_2d_1(x_2d)       # [batch, 32, H, W]
        pool2d_1 = self.maxpool_2d(conv2d_1)  # [batch, 32, H/2, W/2]
        
        conv2d_2 = self.bn_2d_2(pool2d_1)   # [batch, 64, H/2, W/2]
        conv2d_2 = self.se_add_2(conv3d_2, conv2d_2)  # [batch, 64, H/2, W/2]
        pool2d_2 = self.maxpool_2d(conv2d_2)  # [batch, 64, H/4, W/4]
        
        conv2d_3 = self.bn_2d_3(pool2d_2)   # [batch, 128, H/4, W/4]
        conv2d_3 = self.se_add_3(conv3d_3, conv2d_3)  # [batch, 128, H/4, W/4]
        pool2d_3 = self.maxpool_2d(conv2d_3)  # [batch, 128, H/8, W/8]
        
        conv2d_4 = self.bn_2d_4(pool2d_3)   # [batch, 256, H/8, W/8]
        conv2d_4 = self.dropout(conv2d_4)
        pool2d_4 = self.maxpool_2d(conv2d_4)  # [batch, 256, H/16, W/16]
        
        conv2d_5 = self.bn_2d_5(pool2d_4)   # [batch, 512, H/16, W/16]
        conv2d_5 = self.dropout(conv2d_5)
        
        # Decoding
        up_1 = self.up_1(conv2d_5)          # [batch, 256, H/8, W/8]
        # 确保尺寸匹配
        if up_1.shape[2:] != conv2d_4.shape[2:]:
            up_1 = ops.interpolate(up_1, size=conv2d_4.shape[2:], mode='bilinear', align_corners=False)
        merge_1 = self.concat([conv2d_4, up_1])  # [batch, 512, H/8, W/8]
        conv2d_6 = self.bn_2d_6(merge_1)    # [batch, 256, H/8, W/8]
        
        up_2 = self.up_2(conv2d_6)          # [batch, 128, H/4, W/4]
        if up_2.shape[2:] != conv2d_3.shape[2:]:
            up_2 = ops.interpolate(up_2, size=conv2d_3.shape[2:], mode='bilinear', align_corners=False)
        merge_2 = self.concat([conv2d_3, up_2])  # [batch, 256, H/4, W/4]
        conv2d_7 = self.bn_2d_7(merge_2)    # [batch, 128, H/4, W/4]
        
        up_3 = self.up_3(conv2d_7)          # [batch, 64, H/2, W/2]
        if up_3.shape[2:] != conv2d_2.shape[2:]:
            up_3 = ops.interpolate(up_3, size=conv2d_2.shape[2:], mode='bilinear', align_corners=False)
        merge_3 = self.concat([conv2d_2, up_3])  # [batch, 128, H/2, W/2]
        conv2d_8 = self.bn_2d_8(merge_3)    # [batch, 64, H/2, W/2]
        
        up_4 = self.up_4(conv2d_8)          # [batch, 32, H, W]
        if up_4.shape[2:] != conv2d_1.shape[2:]:
            up_4 = ops.interpolate(up_4, size=conv2d_1.shape[2:], mode='bilinear', align_corners=False)
        merge_4 = self.concat([conv2d_1, up_4])  # [batch, 64, H, W]
        conv2d_9 = self.bn_2d_9(merge_4)    # [batch, 32, H, W]
        
        output = self.final_conv(conv2d_9)  # [batch, 1, H, W]
        
        return output

if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    model = DUnet_ms(in_channels_2d=4, in_channels_3d=1)
    
    BATCH_SIZE = 4
    H, W = 192, 192
    
    # 2D输入 (例如：多通道MRI切片)
    x_2d = ms.Tensor(shape=(BATCH_SIZE, 4, H, W), dtype=ms.float32, 
                     init=ms.common.initializer.Normal())
    
    # 3D输入 (可选，如果有真实的3D数据)
    # x_3d = ms.Tensor(shape=(BATCH_SIZE, 1, 8, H, W), dtype=ms.float32, 
    #                  init=ms.common.initializer.Normal())
    
    output = model(x_2d)  # 自动生成3D输入
    # output = model(x_2d, x_3d)  # 如果有真实3D输入
    
    print(f"Output shape: {output.shape}")
    
    y_true = ms.Tensor(shape=(BATCH_SIZE, 1, H, W), dtype=ms.float32,
                      init=ms.common.initializer.Uniform())
    loss = enhanced_mixing_loss(y_true, output)
    print(f"Loss: {loss}")
        