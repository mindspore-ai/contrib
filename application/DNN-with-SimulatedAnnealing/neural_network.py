import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

class ResidualBlock(nn.Cell):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Dense(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.layer2 = nn.Dense(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        # 若 in_features 与 out_features 不同，则使用 Dense 进行匹配，否则使用 Identity
        self.shortcut = nn.Dense(in_features, out_features) if in_features != out_features else ops.Identity()

    def construct(self, x):
        identity = self.shortcut(x)
        out = self.bn1(self.layer1(x))
        out = self.relu(out)
        out = self.bn2(self.layer2(out))
        out += identity
        out = self.relu(out)
        return out

class DeeperResidualNN(nn.Cell):
    def __init__(self):
        super(DeeperResidualNN, self).__init__()
        self.input_layer = nn.Dense(2, 128)
        self.bn_input = nn.BatchNorm1d(128)
        
        self.residual_block1 = ResidualBlock(128, 256)
        self.residual_block2 = ResidualBlock(256, 256)
        self.residual_block3 = ResidualBlock(256, 256)
        self.residual_block4 = ResidualBlock(256, 256)
        self.residual_block5 = ResidualBlock(256, 128)
        
        self.output_layer = nn.Dense(128, 1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.bn_input(self.input_layer(x))
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.output_layer(x)
        return x
