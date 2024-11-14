import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import XavierUniform

class MLP(nn.Cell):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim, act=None,
                 init_fun=XavierUniform()):
        super(MLP, self).__init__()

        layers = []
        for i in range(n_layers):
            dense = nn.Dense(
                in_channels=in_dim if i == 0 else hidden_dim,
                out_channels=out_dim if i == n_layers - 1 else hidden_dim,
                weight_init=init_fun,
                has_bias=True
            )
            layers.append(dense)

            if i < n_layers - 1 and act is not None:
                if isinstance(act, type):
                    layers.append(act())
                else:
                    layers.append(act)

        self._seq = nn.SequentialCell(layers)

    def construct(self, x):
        return self._seq(x)
