import mindspore
from mindspore import nn
import mindspore.ops as ops


class ConditionalLinear(nn.Cell):
    def __init__(self, in_features, out_features, cond_features, method='weak'):
        super(ConditionalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cond_features = cond_features
        self.method = method

        if method == 'weak':
            self.linear_weak = nn.Dense(self.in_features + self.cond_features, self.out_features)
        elif method == 'strong':
            self.linear = nn.Dense(self.in_features, self.out_features)
            self.linear_embedding = nn.Dense(self.cond_features, self.out_features, has_bias=False)
        elif method == 'pure':
            self.bilinear = nn.BiDense(self.in_features, self.cond_features, self.out_features)
        else:
            raise ValueError("Unknown method, should be 'weak', 'strong', or 'pure'.")

    def construct(self, f_in, cond_vec):
        if self.method == 'weak':
            f_out = self.linear_weak(ops.Concat(1)([f_in, cond_vec]))
        elif self.method == 'strong':
            f_out = self.linear(f_in) * self.linear_embedding(cond_vec)
        elif self.method == 'pure':
            f_out = self.bilinear(f_in, cond_vec)
        return f_out
