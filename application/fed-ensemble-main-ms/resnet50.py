from typing import Type, Union, List, Optional
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from download import download
# 初始化卷积层与BatchNorm的参数
# 使用正态分布初始化权重，以0为均值，0.02为标准差
# 初始化卷积层与BatchNorm的参数
weight_init = Normal(mean=0, sigma=0.02)
# 使用正态分布初始化gamma参数，以1为均值，0.02为标准差
gamma_init = Normal(mean=1, sigma=0.02)

class ResidualBlockBase(nn.Cell):
    """
    Residual块的基类。

    Residual块是深度学习模型中的一种结构，用于构建深度神经网络。该类定义了Residual块的基本结构，
    包括两个卷积层、批量归一化和激活函数。子类可以通过重写构造函数和construct方法来定制具体的Residual块。

    Attributes:
        expansion: int，扩展率，用于计算输出通道数。
    """
    expansion: int = 1  # 最后一个卷积核数量与第一个卷积核数量相等

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        """
        初始化ResidualBlockBase。

        参数:
            in_channel (int): 输入通道数。
            out_channel (int): 输出通道数。
            stride (int): 卷积的步长，默认为1。
            norm (Optional[nn.Cell]): 批量归一化层，默认为None。
            down_sample (Optional[nn.Cell]): 下采样层，默认为None。
        """
        super(ResidualBlockBase, self).__init__()
        # 如果没有提供批量归一化层，则使用默认的BatchNorm2d层
        if not norm:
            self.norm = nn.BatchNorm2d(out_channel)
        else:
            self.norm = norm
        # 初始化第一个卷积层
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=weight_init)
        # 初始化第二个卷积层
        self.conv2 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, weight_init=weight_init)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """
        构建Residual块的前向传播。

        参数:
            x: 输入数据。

        返回:
            经过Residual块处理后的输出数据。
        """
        """ResidualBlockBase construct."""
        identity = x  # shortcuts分支

        out = self.conv1(x)  # 主分支第一层：3*3卷积层
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.norm(out)

        # 如果存在下采样层，则对输入进行下采样
        if self.down_sample is not None:
            identity = self.down_sample(x)
        # 将主分支输出与Shortcut（输入）相加
        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out


# 定义ResidualBlock类，用于实现残差块
class ResidualBlock(nn.Cell):
    # 定义扩张率，用于计算输出通道数
    expansion = 4  # 最后一个卷积核的数量是第一个卷积核数量的4倍

    # 初始化函数，设置输入通道数、输出通道数、步长和下采样函数
    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()

        # 初始化第一个1x1卷积层，用于减少输入通道数
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, weight_init=weight_init)
        # 初始化第一个批量归一化层
        self.norm1 = nn.BatchNorm2d(out_channel)
        # 初始化第二个3x3卷积层，用于增大特征图尺寸
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=weight_init)
        # 初始化第二个批量归一化层
        self.norm2 = nn.BatchNorm2d(out_channel)
        # 初始化第三个1x1卷积层，用于增加输出通道数
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion,
                               kernel_size=1, weight_init=weight_init)
        # 初始化第三个批量归一化层
        self.norm3 = nn.BatchNorm2d(out_channel * self.expansion)

        # 初始化激活函数为ReLU
        self.relu = nn.ReLU()
        # 初始化下采样函数，用于当输入和输出尺寸不同时进行下采样
        self.down_sample = down_sample

    # 构造函数，输入特征图x，输出残差学习后的特征图
    def construct(self, x):

        # 初始化identity为输入x，用于残差学习
        identity = x  # shortscuts分支

        # 经过第一个卷积层和批量归一化层
        out = self.conv1(x)  # 主分支第一层：1*1卷积层
        out = self.norm1(out)
        out = self.relu(out)
        # 经过第二个卷积层和批量归一化层
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.norm2(out)
        out = self.relu(out)
        # 经过第三个卷积层和批量归一化层
        out = self.conv3(out)  # 主分支第三层：1*1卷积层
        out = self.norm3(out)

        # 如果存在下采样函数，则对输入x进行下采样
        if self.down_sample is not None:
            identity = self.down_sample(x)

        # 将主分支输出和identity相加，实现残差学习
        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        # 返回残差学习后的特征图
        return out


def make_layer(last_out_channel, block: Type[Union[ResidualBlockBase, ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    """
    创建一个由多个残差块组成的层。

    该函数根据输入参数构建一个包含多个相同类型残差块的层。如果当前层的步长不为1或输入通道数与输出通道数不同，
    则会添加一个下采样层。下采样层用于将输入特征图的尺寸调整到与残差块输出相同的尺寸，以保持维度匹配。

    参数:
    last_out_channel: 前一层的输出通道数。
    block: 残差块的类型，可以是ResidualBlockBase或ResidualBlock。
    channel: 当前层的输出通道数。
    block_nums: 当前层中残差块的数量。
    stride: 当前层的步长，默认为1。

    返回:
    一个由多个残差块组成的SequentialCell实例。
    """

    down_sample = None  # shortcuts分支

    # 根据步长和通道数是否变化决定是否需要添加下采样层
    if stride != 1 or last_out_channel != channel * block.expansion:
        # 使用1x1卷积和批量归一化创建下采样层
        down_sample = nn.SequentialCell([
            nn.Conv2d(last_out_channel, channel * block.expansion,
                      kernel_size=1, stride=stride, weight_init=weight_init),
            nn.BatchNorm2d(channel * block.expansion, gamma_init=gamma_init)
        ])

    # 初始化第一个残差块，可能包含下采样
    layers = []
    layers.append(block(last_out_channel, channel, stride=stride, down_sample=down_sample))

    # 更新输入通道数为当前层的输出通道数
    in_channel = channel * block.expansion
    # 堆叠剩余的残差块
    # 堆叠残差网络
    for _ in range(1, block_nums):
        layers.append(block(in_channel, channel))

    # 返回包含所有残差块的序列模型
    return nn.SequentialCell(layers)


from mindspore import load_checkpoint, load_param_into_net


# 定义ResNet网络类
class ResNet(nn.Cell):
    """
    ResNet网络类。

    ResNet（Residual Network）是一种深度神经网络结构，通过引入残差连接解决了深度网络中的梯度消失和爆炸问题。

    参数:
    - block: 残差块的类型，可以是ResidualBlockBase或ResidualBlock。
    - layer_nums: 每个阶段（layer）的残差块数量列表。
    - num_classes: 分类的类别数量。
    - input_channel: 输入通道的数量。
    """

    def __init__(self, block: Type[Union[ResidualBlockBase, ResidualBlock]],
                 layer_nums: List[int], num_classes: int, input_channel: int) -> None:
        super(ResNet, self).__init__()


        # 初始化第一个卷积层
        # 第一个卷积层，输入channel为3（彩色图像），输出channel为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=weight_init)
        # 初始化批量归一化层
        self.bn1 = nn.BatchNorm2d(64)
        # 引入ReLU激活函数
        self.relu = nn.ReLU()
        # 初始化最大池化层
        # 最大池化层，缩小图片的尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        # 通过make_layer函数构建网络的四个主要层
        # 各个残差网络结构块定义
        self.layer1 = make_layer(64, block, 64, layer_nums[0])
        self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
        self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)

        # 初始化全局平均池化层
        # 平均池化层
        self.avgpool = nn.AvgPool2d()
        # 初始化展平层，用于将二维特征图展平为一维向量
        # flattern层
        self.flatten = nn.Flatten()
        # 初始化全连接层，用于分类
        # 全连接层
        self.fc = nn.Dense(in_channels=input_channel, out_channels=num_classes)

    # 定义前向传播方法
    def construct(self, x):
        """
        前向传播方法。

        参数:
        - x: 输入数据。

        返回:
        - 输出数据。
        """
        # 经过第一个卷积层
        x = self.conv1(x)
        # 经过批量归一化层
        x = self.norm(x)
        # 经过ReLU激活函数
        x = self.relu(x)
        # 经过最大池化层
        x = self.max_pool(x)

        # 经过四个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 经过全局平均池化层
        x = self.avg_pool(x)
        # 展平特征图
        x = self.flatten(x)
        # 经过全连接层
        x = self.fc(x)

        return x

def _resnet(model_url: str, block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int], num_classes: int, pretrained: bool, pretrained_ckpt: str,
            input_channel: int):
    """
    构建ResNet模型。

    参数:
    model_url: 模型下载地址。
    block: ResNet中的残差块类型。
    layers: 每个阶段的残差块数量列表。
    num_classes: 输出类别数量。
    pretrained: 是否使用预训练模型。
    pretrained_ckpt: 预训练模型的存储路径。
    input_channel: 输入通道数量。

    返回:
    构建好的ResNet模型。
    """
    # 初始化ResNet模型
    model = ResNet(block, layers, num_classes, input_channel)

    # 如果需要预训练模型
    if pretrained:
        # 加载预训练模型
        # 加载预训练模型
        download(url=model_url, path=pretrained_ckpt, replace=True)
        param_dict = load_checkpoint(pretrained_ckpt)
        # 将预训练参数加载到模型中
        load_param_into_net(model, param_dict)

    return model


def resnet50(num_classes: int = 1000, pretrained: bool = False):
    """
    构建ResNet-50模型。

    参数:
    num_classes: 输出类别数量，默认为1000。
    pretrained: 是否使用预训练模型，默认为False。

    返回:
    构建好的ResNet-50模型。
    """
    """ResNet50模型"""
    # 定义ResNet-50的预训练模型地址和存储路径
    resnet50_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/models/application/resnet50_224_new.ckpt"
    resnet50_ckpt = "./LoadPretrainedModel/resnet50_224_new.ckpt"
    # 调用_resnet函数构建ResNet-50模型
    return _resnet(resnet50_url, ResidualBlock, [3, 4, 6, 3], num_classes,
                   pretrained, resnet50_ckpt, 2048)
