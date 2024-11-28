import mindspore as ms
from mindspore import nn, ops

# refer to https://arxiv.org/pdf/1612.01887.pdf

HIDDEN_DIM = 512
N_IMAGE_DESCRIPTOR = 49
IMAGE_RAW_DIM = 2048
EMB_DIM = 512
N_VOCAB = 40 * 1000 # for caption; 60 * 1000 for hashtag

ms.set_context(device_target = "Ascend" if ms.hal.is_available("Ascend") else "cpu")

class AdaptiveAttention(nn.Cell):
    def __init__(self):
        super(AdaptiveAttention, self).__init__()
        self.w_h = ms.Parameter(ops.randn(N_IMAGE_DESCRIPTOR))
        self.W_v = ms.Parameter(ops.randn((HIDDEN_DIM, N_IMAGE_DESCRIPTOR)))
        self.W_g = ms.Parameter(ops.randn((HIDDEN_DIM, N_IMAGE_DESCRIPTOR)))
        self.W_s = ms.Parameter(ops.randn((HIDDEN_DIM, N_IMAGE_DESCRIPTOR)))

    # V; image descriptor , s_t : visual sentinel, h_t: output(hidden) of the lstm
    def construct(self, V, s_t, h_t):
        # eq (6)
        image_part = ops.matmul(V, self.W_v) # V -1, 49, 2048 / W_v 2048, 49/ output -1, 49, 49
        single_hidden_part = ops.matmul(h_t, self.W_g) # h_t -1, 512/ W_g 512, 49 output -1, 49
        dummy_1 = ops.ones((single_hidden_part.shape[0], 1, N_IMAGE_DESCRIPTOR)) # -1, 1, 49
        hidden_part = ops.bmm(single_hidden_part.unsqueeze(2), dummy_1) # -1, 49, 1 bmm -1, 1, 49 output -1, 49, 49
        z_t = ops.matmul(ops.tanh(image_part + hidden_part), self.w_h)

        # eq (7)
        alpha_t = ops.softmax(z_t, axis = 1)

        # eq(8)
        c_t = ops.sum(V * alpha_t.unsqueeze(2), dim = 1)

        # eq(12)

        attention_vs = ops.matmul(ops.tanh(ops.matmul(s_t, self.W_s) + single_hidden_part), self.w_h)
        concatenates = ops.concat([z_t, attention_vs.unsqueeze(1)], axis = 1)
        alpha_t_hat = ops.softmax(concatenates, axis = 1)

        # beta_t = alpha_t[k+1] , last element of the alpha_t
        beta_t = alpha_t_hat[:,-1:]
        # eq(11)
        c_t_hat = beta_t * s_t + (1-beta_t) * c_t
        return c_t_hat

class ExtendedLSTM(nn.Cell):
    def __init__(self, input_size, output_size):
        super(ExtendedLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size = input_size, hidden_size = output_size)
        self.sentinel = VisualSentinel()
        # x_t is [w_t; v_g]
    def construct(self, x_t, prev_hidden, prev_cell_state):
        (curr_hidden , curr_cell) = self.lstm(x_t, (prev_hidden, prev_cell_state))
        s_t = self.sentinel(x_t, prev_hidden, curr_cell)
        return s_t, curr_hidden, curr_cell

class VisualSentinel(nn.Cell):
    def __init__(self):
        super(VisualSentinel, self).__init__()
        # since x_t is [w_t; v_g]
        self.W_x = ms.Parameter(ops.randn((HIDDEN_DIM + EMB_DIM, HIDDEN_DIM)), name = "W_x")
        self.W_h = ms.Parameter(ops.randn((HIDDEN_DIM, HIDDEN_DIM)), name = "W_h")

    # m_t is the lstm cell state
    def construct(self, x_t, prev_h_t, m_t):
        # eq (9) / (10)
        g_t = ops.sigmoid(ops.matmul(x_t, self.W_x) + ops.matmul(prev_h_t, self.W_h)) # output -1, 512, 512
        s_t = ops.mul(g_t, ops.tanh(m_t))
        return s_t

class AdaptiveAttentionLSTMNetwork(nn.Cell):
    def __init__(self, input_size, output_size):
        super(AdaptiveAttentionLSTMNetwork, self).__init__()
        self.attention = AdaptiveAttention()
        self.extended_lstm = ExtendedLSTM(input_size = input_size, output_size = output_size)
        self.mlp = nn.Dense(HIDDEN_DIM, N_VOCAB, has_bias = False)
    # x_t : [w_t; v_g] / V = [v_1, v_2, ... v_49]
    def construct(self, V, x_t, prev_hidden, prev_cell_state):
        s_t, curr_hidden, curr_cell = self.extended_lstm(x_t, prev_hidden, prev_cell_state)
        c_t_hat = self.attention(V, s_t, curr_hidden)
        # output = F.softmax(self.mlp(c_t_hat + curr_hidden),dim=1)
        output = self.mlp(c_t_hat + curr_hidden) # use X-entrophy loss!
        return output, curr_hidden, curr_cell


def main():
    BATCH_SIZE = 32
    V = ops.randn((BATCH_SIZE, N_IMAGE_DESCRIPTOR, HIDDEN_DIM))
    x_t = ops.randn((BATCH_SIZE, HIDDEN_DIM + EMB_DIM))
    h_0, c_0 = get_start_states(batch_size = BATCH_SIZE)
    model = AdaptiveAttentionLSTMNetwork(input_size = HIDDEN_DIM + EMB_DIM, output_size = HIDDEN_DIM)
    output, curr_hidden, curr_cell = model(V, x_t, h_0, c_0)
    print(output)
    print(curr_hidden)
    print(curr_cell)
    print("test done ...")

def get_start_states(batch_size):
    hidden_dim = HIDDEN_DIM
    h0 = ops.zeros((batch_size, hidden_dim))
    c0 = ops.zeros((batch_size, hidden_dim))
    return h0, c0

if __name__ == '__main__':
    main()
