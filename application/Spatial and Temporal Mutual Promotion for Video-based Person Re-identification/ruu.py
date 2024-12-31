import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class BasicConv2d(nn.Cell):
    """带批量归一化和ReLU激活的基础卷积块"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, has_bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        """卷积块的前向传播"""
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class BasicConv2dLeaky(nn.Cell):
    """带批量归一化和LeakyReLU激活的基础卷积块"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2dLeaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, has_bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.leaky_relu = nn.LeakyReLU(alpha=0.1)

    def construct(self, x):
        """卷积块的前向传播"""
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class RRU(nn.Cell):
    """用于视频处理的循环残差单元"""

    def __init__(self, input_size):
        super().__init__()
        self.hidden_size = 256

        bottleneck_size = [256, 128]
        self.reduce_dim_z = BasicConv2d(
            input_size * 2,
            bottleneck_size[0],
            kernel_size=1,
            pad_mode='valid'
        )
        self.s_atten_z = nn.SequentialCell([
            nn.Conv2d(1, bottleneck_size[1], kernel_size=3,
                     pad_mode='same', has_bias=False),
            nn.ReLU(),
            nn.Conv2d(bottleneck_size[1], 64, kernel_size=1,
                     pad_mode='valid', has_bias=False)
        ])
        self.c_atten_z = nn.SequentialCell([
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(bottleneck_size[0], input_size,
                     kernel_size=1, pad_mode='valid', has_bias=False)
        ])
        self.sigmoid = nn.Sigmoid()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.concat = ops.Concat(axis=1)
        self.stack = ops.Stack(axis=2)

    def generate_attention_z(self, x):
        """为输入特征生成注意力图"""
        z = self.reduce_dim_z(x)
        atten_s = self.s_atten_z(self.mean(z, 1))
        atten_c = self.c_atten_z(z)
        z = self.sigmoid(atten_s * atten_c)
        return z, 1 - z

    def construct(self, x):
        """RRU模型的前向传播"""
        if len(x.shape) == 4:
            x = ops.ExpandDims()(x, 0)

        depth = x.shape[1]

        first_frame = ops.ExpandDims()(x[:, 0], 1)
        res = self.concat((first_frame, x))
        res = res[:, :-1]
        res = x - res

        h = x[:, 0]
        output = []
        for t in range(depth):
            con_fea = self.concat((h - x[:, t], res[:, t]))
            z_p, z_r = self.generate_attention_z(con_fea)
            h = z_r * h + z_p * x[:, t]
            output.append(h)

        fea = self.stack(output)
        return fea


def main():
    """主函数：用于演示RRU模型的详细行为"""
    # 设置运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    # 创建测试数据
    batch_size = 2
    time_steps = 4
    channels = 64
    height = 32
    width = 32

    # 创建随机输入数据
    input_data = np.random.rand(
        batch_size, time_steps, channels, height, width
    ).astype(np.float32)
    input_tensor = Tensor(input_data)

    # 初始化模型
    model = RRU(input_size=channels)

    # 前向传播
    output = model(input_tensor)

    # 详细输出信息
    print("模型信息:")
    print("-" * 50)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    
    # 检查输出值的范围
    output_np = output.asnumpy()
    print(f"\n输出统计信息:")
    print("-" * 50)
    print(f"最小值: {output_np.min():.4f}")
    print(f"最大值: {output_np.max():.4f}")
    print(f"平均值: {output_np.mean():.4f}")
    print(f"标准差: {output_np.std():.4f}")
    
    print("\n模型运行成功!")


if __name__ == '__main__':
    main()
