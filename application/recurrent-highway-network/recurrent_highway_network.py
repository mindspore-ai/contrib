import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from typing import List
from mindspore import Tensor

class RHNCell(nn.Cell):
    __constants__ = ['nb_rhn_layers', 'drop_prob', 'hidden_dim']

    def __init__(self, input_dim, hidden_dim, nb_rhn_layers, drop_prob):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nb_rhn_layers = nb_rhn_layers
        self.drop_prob = drop_prob

        self.drop_layer = nn.Dropout(p=drop_prob)
        self.input_fc = nn.Dense(input_dim, 2 * hidden_dim)
        self.first_fc_layer = nn.Dense(hidden_dim, 2 * hidden_dim)
        self.fc_layers = nn.SequentialCell([nn.Dense(hidden_dim, 2 * hidden_dim) for _ in range(nb_rhn_layers-1)])

    def highwayGate(self, hidden_states, s_t_l_minus_1):
        h, t = ops.split(hidden_states, self.hidden_dim, 1)
        h, t = ops.tanh(h), ops.sigmoid(t)
        c = 1 - t
        t = self.drop_layer(t)
        s_t = h * t + s_t_l_minus_1 * c
        return s_t

    def construct(self, x_t, s_t_l_0):

        hidden_states = self.input_fc(x_t) + self.first_fc_layer(s_t_l_0)
        s_t_l = self.highwayGate(hidden_states, s_t_l_0)

        s_t_l_minus_1 = s_t_l
        for fc_layer in self.fc_layers:
            hidden_states = fc_layer(s_t_l_minus_1)
            s_t_l = self.highwayGate(hidden_states, s_t_l_minus_1)
            s_t_l_minus_1 = s_t_l

        return s_t_l


class RHN(nn.Cell):

    def __init__(self, input_dim, hidden_dim, nb_rhn_layers, drop_prob):
        super().__init__()

        self.rhncell = RHNCell(input_dim, hidden_dim, nb_rhn_layers, drop_prob)
        self.output_fc = nn.Dense(hidden_dim, hidden_dim)

    def construct(self, input, s_t_0_l_0):

        inputs = input.unbind(1)
        s_t_minus_1_L = s_t_0_l_0
        
        outputs = []
        for t in range(len(inputs)):
            s_t_L = self.rhncell(inputs[t], s_t_minus_1_L)
            s_t_minus_1_L = s_t_L
            outputs += [s_t_L]
        print(ops.stack(outputs).shape)
        x = ops.stack(outputs).transpose(1, 0, 2)
        x = self.output_fc(x)
        return x
