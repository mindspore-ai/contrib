import mindspore as ms
from mindspore import nn, ops
import numpy as np

class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, norm_type="none"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.CellList()
        self.norms = nn.CellList()
        self.norm_type = norm_type

        if num_layers == 1:
            self.layers.append(nn.Dense(input_dim, output_dim))
        else:
            self.layers.append(nn.Dense(input_dim, hidden_dim))
            if self.norm_type == "bn":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "ln":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for _ in range(num_layers - 2):
                self.layers.append(nn.Dense(hidden_dim, hidden_dim))
                if self.norm_type == "bn":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "ln":
                    self.norms.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.Dense(hidden_dim, output_dim))

    def construct(self, feats):
        h = feats
        for (l, layer) in enumerate(self.layers):
            h = layer(h)
            if (l != self.num_layers - 1):
                h = self.dropout(h)
                h = ops.ReLU()(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
        return h

class SAMLP(nn.Cell):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_nodes, num_layer=1, dropout=.5,
                 norm_type='none', use_predictor=False):
        super().__init__()
        self.mlpA = nn.Dense(num_nodes, hidden_channels)
        self.mlpX = nn.Dense(in_channels, hidden_channels)
        self.atten = nn.Dense(hidden_channels * 2, 1)
        self.classifierA = MLP(hidden_channels, hidden_channels, out_channels,
                               num_layer, dropout, norm_type=norm_type)
        self.classifierX = MLP(hidden_channels, hidden_channels, out_channels,
                               num_layer, dropout, norm_type=norm_type)
        self.dropout = nn.Dropout(p=dropout)
        self.use_predictor = use_predictor
        self.latent_predictor = MLP(in_channels, hidden_channels, hidden_channels,
                                    2, dropout, norm_type=norm_type)

    def decouple_encoder(self, A, X):
        if self.use_predictor:
            HA = self.latent_predictor(X)
        else:
            HA = self.mlpA(A)
        HA = self.dropout(HA)
        HA = ops.ReLU()(HA)

        HX = self.mlpX(X)
        HX = self.dropout(HX)
        HX = ops.ReLU()(HX)

        H = ops.Concat(1)([HA, HX])
        return H, HA, HX

    def attentive_decoder(self, H, HA, HX):
        yA = self.classifierA(HA)
        yX = self.classifierX(HX)

        alpha = ops.Sigmoid()(self.atten(H))
        alpha_reshaped = ops.Reshape()(alpha, (-1, 1))
        y = yA * alpha_reshaped + yX * (1 - alpha_reshaped)
        return y

    def construct(self, A, X):
        H, HA, HX = self.decouple_encoder(A, X)
        y = self.attentive_decoder(H, HA, HX)
        return y
if __name__ == '__main__':
    in_channels = 10
    hidden_channels = 32
    out_channels = 5
    num_nodes = 20
    num_layers = 2
    dropout = 0.5
    norm_type = 'bn'
    use_predictor = True
    model = SAMLP(in_channels, hidden_channels, out_channels, num_nodes,
                  num_layers, dropout, norm_type, use_predictor)
    A = ms.Tensor(np.random.rand(num_nodes, num_nodes).astype(np.float32))
    X = ms.Tensor(np.random.rand(num_nodes, in_channels).astype(np.float32))
    print(model(A, X))