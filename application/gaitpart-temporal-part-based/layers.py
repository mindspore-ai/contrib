import mindspore
from mindspore import nn, ops
import copy

def clones(Cell, N):
    "Produce N identical layers."
    return nn.CellList([copy.deepcopy(Cell) for _ in range(N)])

class BasicConv1d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, has_bias=False, **kwargs)

    def construct(self, x):
        ret = self.conv(x)
        return ret

class FocalConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super().__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, has_bias=False, **kwargs)

    def construct(self, x):
        h = x.size(2)
        split_size = int(h // 2**self.halving)
        z = x.split(split_size, 2)
        z = ops.cat([self.conv(_) for _ in z], 2)
        return ops.leaky_relu(z, inplace=True)

class TemporalFeatureAggregator(nn.Cell):
    def __init__(self, in_channels, squeeze=4, part_num=16):
        super().__init__()
        hidden_dim = int(in_channels // squeeze)
        self.part_num = part_num

        # MTB1
        conv3x1 = nn.SequentialCell(
                BasicConv1d(in_channels, hidden_dim, 3),
                nn.LeakyReLU(),
                BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, part_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1, pad_mode='pad')
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1, pad_mode='pad')

        # MTB1
        conv3x3 = nn.SequentialCell(
                BasicConv1d(in_channels, hidden_dim, 3),
                nn.LeakyReLU(),
                BasicConv1d(hidden_dim, in_channels, 3))
        self.conv1d3x3 = clones(conv3x3, part_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2, pad_mode='pad')
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2, pad_mode='pad')

    def construct(self, x):
        """
          Input: x, [p, n, c, s]
        """
        p, n, c, s = x.shape
        feature = x.split(1, 0)
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = ops.cat([conv(_.squeeze(0)).expand_dims(0)
            for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = ops.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = ops.cat([conv(_.squeeze(0)).expand_dims(0)
            for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = ops.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = (feature3x1 + feature3x3).max(-1)[0]
        return ret

if __name__ == '__main__':
    x = ops.ones((1, 1, 64, 128))
    model = TemporalFeatureAggregator(in_channels=64, squeeze=4, part_num=16)
    output = model(x)
    print(output)
    print(f'output.shape={output.shape}')