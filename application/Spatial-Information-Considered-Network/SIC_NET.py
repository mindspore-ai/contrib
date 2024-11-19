import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Normal, XavierUniform
import numpy as np


class ResNet_block(nn.Cell):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=3,
                 padding=1):
        super(ResNet_block, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(input_channels),
            nn.Tanh()
        ])
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(output_channels),
            nn.Tanh()
        ])
        self.bn = nn.SequentialCell([
            nn.BatchNorm2d(output_channels),
            nn.Tanh()
        ])

    def construct(self, x):
        input_ = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.bn(x + input_)


class ReNet_layer(nn.Cell):
    def __init__(self, input_channels, output_channels):
        super(ReNet_layer, self).__init__()
        if output_channels % 2 != 0:
            output_channels += 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        hidden_size_v = int(input_channels / 2)
        hidden_size_h = int(output_channels / 2)

        self.Vlayer = nn.GRU(input_size=input_channels,
                             hidden_size=hidden_size_v,
                             num_layers=1,
                             has_bias=True,
                             bidirectional=True,
                             batch_first=False)

        self.Hlayer = nn.GRU(input_size=input_channels,
                             hidden_size=hidden_size_h,
                             num_layers=1,
                             has_bias=True,
                             bidirectional=True,
                             batch_first=False)

    def construct(self, x):
        b, c, m, n = x.shape
        x = ops.Transpose()(x, (2, 3, 0, 1))  # (m, n, b, c)
        V_map = []
        for i in range(n):
            input_seq = x[:, i, :, :]  # (m, b, c)
            output, _ = self.Vlayer(input_seq)
            V_map.append(output)
        V_map = ops.Stack(axis=1)(V_map)  # (m, n, b, hidden_size*2)

        H_map = []
        for i in range(m):
            input_seq = V_map[i, :, :, :]  # (n, b, hidden_size*2)
            output, _ = self.Hlayer(input_seq)
            H_map.append(output)
        H_map = ops.Stack(axis=0)(H_map)  # (m, n, b, output_channels)

        H_map = ops.Transpose()(H_map, (2, 3, 0, 1))  # (b, output_channels, m, n)
        return H_map


class Re_CNN_block(nn.Cell):
    def __init__(self, input_channels, output_channels):
        super(Re_CNN_block, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        ])

        self.renet1 = ReNet_layer(input_channels, input_channels)

        self.conv2 = nn.SequentialCell([
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        ])

        self.renet2 = ReNet_layer(input_channels, output_channels)

        self.bn = nn.SequentialCell([
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        ])

    def construct(self, x):
        input_ = x
        x1 = self.conv1(x)
        x1 = self.renet2(x1)
        x2 = self.renet1(x)
        x2 = self.conv2(x2)
        return self.bn(x1 + x2 + input_)


class model_sic_net(nn.Cell):
    def __init__(self,
                 input_feature_map_channels=1024,
                 num_classes=7,
                 num_CTU=3,
                 num_FCN_layers=3):
        '''
        input_feature_map_channels: 编码结果的维度
        num_classes: 类别总数
        num_CTU: crossing transfer unit的数量
        num_FCN_layers: 最后的全卷积层数量（包含分类层）
        '''
        super(model_sic_net, self).__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.SequentialCell([
            nn.BatchNorm2d(input_feature_map_channels),
            nn.Tanh(),
            nn.Conv2d(input_feature_map_channels, 512, kernel_size=1, stride=1, pad_mode='valid'),
            nn.BatchNorm2d(512),
            nn.Tanh()
        ])
        CTU_layers = []
        for i in range(num_CTU):
            CTU_layers.append(Re_CNN_block(512, 512))
        self.CTU = nn.SequentialCell(CTU_layers)
        self.bn = nn.SequentialCell([
            nn.BatchNorm2d(512),
            nn.Tanh()
        ])

        FCN_layers = []
        for i in range(num_FCN_layers - 1):
            FCN_layers.append(nn.SequentialCell([
                nn.Dropout2d(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
                nn.BatchNorm2d(512),
                nn.Tanh()
            ]))
        self.fcn = nn.SequentialCell(FCN_layers)

        self.out = nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=1, pad_mode='pad')

    def construct(self, x):
        x = self.conv1x1(x)
        input_ = x
        x = self.CTU(x)
        x = self.bn(x + input_)
        x = self.fcn(x)
        return self.out(x)


if __name__ == "__main__":
    import mindspore.context as context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    input_tensor = Tensor(np.random.rand(4, 512, 32, 32), mindspore.float32)
    model = model_sic_net(512, 7, 3, 3)
    output = model(input_tensor)
    print(output.shape)