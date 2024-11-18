# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import mindspore.nn as nn
import mindspore.ops as ops

class RNN(nn.Cell):
    """
    Implementation of recurrent neural network using
    `nn.Linear` class following "Generating Sequences With 
    Recurrent Neural Networks" by Alex Graves.

    There are 4 types of layer connections:
    - input to all hidden layers
    - hiddens to all next layer hidden layers
    - all previous time step hiddens to all current time step hiddens
    - hiddens to all outputs

    Note that the `hidden_size` is same for all layers.
    
    TODO as follows:
      - Add sequence of inputs producing sequence of corresponding
        outputs after forward pass.
      - Make this RNN bidirectional.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(RNN, self).__init__()
        # Set the sizes of layers and more.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #self.sequence_length = sequence_length
        self.num_layers = num_layers

        self.ih = []    # input to hidden (all) like skip connections
        self.hh = []    # current hidden to next layer hidden
        self._hh = []   # previous in time hidden to current hidden
        if bidirectional:   # TODO
            self.hh_ = []   # future time hidden to current hidden
        self.ho = []    # all hidden layers to output connections

        # Initialize all layers.
        for _ in range(self.num_layers):
            self.ih.append(nn.Dense(self.input_size, self.hidden_size))
            self._hh.append(nn.Dense(self.hidden_size, self.hidden_size))
            self.ho.append(nn.Dense(self.hidden_size, self.output_size))
        for _ in range(1, self.num_layers):  # connections are one less than number of layers
            self.hh.append(nn.Dense(self.hidden_size, self.hidden_size))

        # Set the lower limit for `num_layers`.
        assert num_layers >= 1, '`num_layers` must be >= 1. Default: 1'

    def construct(self, x, h_states):
        """Forward pass corresponds to current time step computation (`x` input)
        taking into account previous time step computation accessed from previous
        time step `h_states`.

        `x`: [batch_size, seq_len, input_size]
        """
        # Compare length of `h_states` to `num_layers`.
        assert len(h_states) == self.num_layers, 'Number of `h_states` must be equal to `num_layers`.'
        # Stare all `h_states` in `stored_h_states`.
        stored_h_states = []

        # Compute for all the hidden layers.
        for layer in range(self.num_layers):
            if layer == 0:  # if this is the first layer
                h1 = ops.tanh(self.ih[layer](x) + self._hh[layer](h_states[layer]))
                stored_h_states.append(h1)
            else:
                hn = ops.tanh(self.ih[layer](x) + self._hh[layer](h_states[layer]) + self.hh[layer - 1](h1))
                stored_h_states.append(hn)

        # Compute the output using all the previously computed hidden layers,
        out = 0
        for layer in range(self.num_layers):
            out += self.ho[layer](stored_h_states[layer])
        return out, stored_h_states

    def __str__(self):
        return 'RNN(input_size={}, hidden_size={}, output_size={}, num_layers={})'.format(
            self.input_size, self.hidden_size, self.output_size, self.num_layers
        )
    
def main():
    input_size = 10  # 输入特征的维度
    hidden_size = 20  # 隐藏层的维度
    output_size = 1  # 输出的维度（例如，对于二分类任务）
    num_layers = 2  # RNN的层数
    batch_size = 32  # 批处理大小
    sequence_length = 5  # 序列长度
    rnn = RNN(input_size, hidden_size, output_size, num_layers)

    print(rnn)

    x = ops.randn((batch_size, sequence_length, input_size))

    h0 = [ops.zeros((batch_size, hidden_size)) for _ in range(num_layers)]
    x_first_timestep = x[:, 0, :]  
    output, h_states = rnn(x_first_timestep, h0)  

    print("Output:", output)
    for layer, h in enumerate(h_states):
        print(f"Hidden state of layer {layer}:", h)
 
if __name__ == "__main__":
    main()