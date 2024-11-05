import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor
from mindspore.common.initializer import Uniform
from mindspore.ops import operations as P

class Patient2Vec(nn.Cell):
    """
    Self-attentive representation learning framework,
    including convolutional embedding layer,
    recurrent autoencoder with an encoder, recurrent module, and a decoder.
    In addition, a linear layer is on top of each decode step, and the weights are shared at these steps.
    """

    def __init__(self, input_size, hidden_size, n_layers, att_dim, initrange,
                 output_size, rnn_type, seq_len, pad_size, n_filters, bi, dropout_p=0.5):
        super(Patient2Vec, self).__init__()

        self.initrange = initrange
                # convolution
        self.b = 1
        if bi:
            self.b = 2

        # Convolutional layers
        self.conv = nn.Conv1d(1, 1, kernel_size=input_size, stride=2, pad_mode='valid')
        self.conv2 = nn.Conv1d(1, n_filters, kernel_size=hidden_size * self.b, stride=2, pad_mode='valid')

        # Recurrent layers
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, dropout=dropout_p,
                                         batch_first=True, bidirectional=bi)

        # Attention layers
        self.att_w1 = nn.Dense(hidden_size * self.b, att_dim, has_bias=False)
        self.linear = nn.Dense(hidden_size * self.b * n_filters + 3, output_size)

        self.func_softmax = nn.Softmax()
        self.func_sigmoid = nn.Sigmoid()
        self.func_tanh = nn.Hardtanh(0, 1)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_weights()

        self.pad_size = pad_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_filters = n_filters

        

    def init_weights(self):
        """
        Initialize weights
        """
        for param in self.get_parameters():
            param = ops.uniform(param.shape, -self.initrange, self.initrange)

    def convolutional_layer(self, inputs):
        convolution_all = []
        conv_wts = []
        for i in range(self.seq_len):
            convolution_one_month = []
            for j in range(self.pad_size):
                convolution = self.conv(ops.unsqueeze(inputs[:, i, j], dim=1))
                convolution_one_month.append(convolution)
            convolution_one_month = ops.stack(convolution_one_month)
            convolution_one_month = ops.squeeze(convolution_one_month, axis=3)
            convolution_one_month = ops.swapaxes(convolution_one_month, 0, 1)
            convolution_one_month = ops.swapaxes(convolution_one_month, 1, 2)
            convolution_one_month = ops.squeeze(convolution_one_month, axis=1)
            convolution_one_month = self.func_tanh(convolution_one_month)
            convolution_one_month = ops.unsqueeze(convolution_one_month, dim=1)
            vec = ops.bmm(convolution_one_month, inputs[:, i])
            convolution_all.append(vec)
            conv_wts.append(convolution_one_month)
        convolution_all = ops.stack(convolution_all, axis=1)
        convolution_all = ops.squeeze(convolution_all, axis=2)
        conv_wts = ops.squeeze(ops.stack(conv_wts, axis=1), axis=2)
        return convolution_all, conv_wts

    def encode_rnn(self, embedding, batch_size):
        init_state = ops.zeros((self.n_layers * self.b, batch_size, self.hidden_size), ms.float32)
        embedding = self.dropout(embedding)
        outputs_rnn, _ = self.rnn(embedding, init_state)
        return outputs_rnn

    def add_beta_attention(self, states, batch_size):
        # beta attention
        att_wts = []
        for i in range(self.seq_len):
            m1 = self.conv2(ops.unsqueeze(states[:, i], dim=1))
            att_wts.append(ops.squeeze(m1, axis=2))
        att_wts = ops.stack(att_wts, axis=2)
        att_beta = []
        for i in range(self.n_filters):
            a0 = self.func_softmax(att_wts[:, i])
            att_beta.append(a0)
        att_beta = ops.stack(att_beta, axis=1)
        context = ops.bmm(att_beta, states)
        context = context.view(batch_size, -1)
        return att_beta, context

    def construct(self, inputs, inputs_other, batch_size):
        # Convolutional
        convolutions, alpha = self.convolutional_layer(inputs)
        # RNN
        states_rnn = self.encode_rnn(convolutions, batch_size)
        # Attention and context vector
        beta, context = self.add_beta_attention(states_rnn, batch_size)
        context_v2 = ops.concat((context, inputs_other), axis=1)
        linear_y = self.linear(context_v2)
        out = self.func_sigmoid(linear_y)
        return out, alpha, beta


def get_loss(pred, y, criterion, mtr, a=0.5):
    mtr_t = ops.transpose(mtr, (0, 2, 1))
    aa = ops.matmul(mtr, mtr_t)
    loss_fn = 0
    for i in range(aa.shape[0]):
        aai = aa[i] - Tensor(np.eye(mtr.shape[1]), ms.float32)
        loss_fn += ops.trace((aai * aai).sum())
    loss_fn /= aa.shape[0]
    loss = criterion(pred, y) + Tensor(loss_fn * a, ms.float32)
    return loss


def main():
    input_size = 5
    hidden_size = 16
    n_layers = 1
    att_dim = 8
    initrange = Tensor(0.1)
    output_size = 3
    rnn_type = 'GRU'
    seq_len = 4
    pad_size = 2
    n_filters = 2
    bi = True
    dropout_p = 0.5

    model = Patient2Vec(input_size, hidden_size, n_layers, att_dim, initrange,
                        output_size, rnn_type, seq_len, pad_size, n_filters, bi, dropout_p)
    batch_size = 3
    inputs = ops.rand(batch_size, seq_len, pad_size, input_size)
    inputs_other = ops.rand(batch_size, 3)
    print('input.shape', inputs.shape)
    output, alpha, beta = model(inputs, inputs_other, batch_size)
    print("模型输出:", output)
    print("Alpha注意力权重:", alpha)
    print("Beta注意力权重:", beta)


if __name__ == "__main__":
    main()