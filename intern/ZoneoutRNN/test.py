import mindspore
from ZoneoutRNN import ZoneoutRNN
import mindspore.nn as nn
import mindspore.ops as ops

batch_size = 10
max_time = 20
feature_dims = 50
hidden_size = 100
zoneout_prob = (0.1, 0.1)
dtype = mindspore.float32

inputs = ops.randn([batch_size, max_time, feature_dims])
forward_cell = nn.LSTMCell(feature_dims, hidden_size)
backward_cell = nn.LSTMCell(feature_dims, hidden_size)
zoneout_rnn = ZoneoutRNN(forward_cell, backward_cell, zoneout_prob)
outputs = ops.zeros([batch_size, max_time, hidden_size*2], dtype=dtype)
forward_h = ops.zeros([batch_size, hidden_size], dtype=dtype)
forward_c = ops.zeros([batch_size, hidden_size], dtype=dtype)
forward_state = (forward_h, forward_c)
backward_h = ops.zeros([batch_size, hidden_size], dtype=dtype)
backward_c = ops.zeros([batch_size, hidden_size], dtype=dtype)
backward_state = (backward_h, backward_c)

for i in range(max_time):
    forward_input = inputs[:, i, :]
    backward_input = inputs[:, max_time-(i+1), :]
    forward_output, backward_output, forward_new_state, backward_new_state = zoneout_rnn(
        forward_input, backward_input, forward_state, backward_state)
    forward_state = forward_new_state
    backward_sate = backward_new_state
    outputs[:, i, :hidden_size] = forward_output
    outputs[:, max_time-(i+1), hidden_size:] = backward_output

print(outputs)