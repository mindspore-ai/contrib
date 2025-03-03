from mindspore import nn, ops, Tensor
import mindspore as ms

class Feature_Encoder(nn.Cell):
    def __init__(self, input_dim: int, output_dim: int = 20):
        super().__init__()
        self.linear = nn.Dense(input_dim, output_dim, 
                              weight_init=ms.common.initializer.XavierUniform(),
                              bias_init='zeros')
        self.relu = nn.ReLU()
    
    def construct(self, x):
        return self.relu(self.linear(x))

class Classifier(nn.Cell):
    def __init__(self, input_dim=20):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(2*input_dim, 1,
                    weight_init=ms.common.initializer.XavierUniform(),
                    bias_init='zeros')
        )

    def construct(self, x):
        return self.net(x)