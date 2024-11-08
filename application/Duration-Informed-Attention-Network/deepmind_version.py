import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context
from mindspore.common.initializer import initializer, Zero


class WaveRNN(nn.Cell):
    def __init__(self, hidden_size=384, quantization=256):
        super(WaveRNN, self).__init__()

        self.hidden_size = hidden_size
        self.split_size = hidden_size // 2

        self.R = nn.Dense(self.hidden_size, 3 * self.hidden_size, has_bias=False)

        self.O1 = nn.Dense(self.split_size, self.split_size)
        self.O3 = nn.Dense(self.split_size, self.split_size)

        self.one_O2 = nn.Dense(self.split_size, quantization)
        self.one_O4 = nn.Dense(self.split_size, quantization)
        self.two_O2 = nn.Dense(self.split_size, quantization)
        self.two_O4 = nn.Dense(self.split_size, quantization)
        self.three_O2 = nn.Dense(self.split_size, quantization)
        self.three_O4 = nn.Dense(self.split_size, quantization)
        self.four_O2 = nn.Dense(self.split_size, quantization)
        self.four_O4 = nn.Dense(self.split_size, quantization)

        self.I_coarse = nn.Dense(2 * 4, 3 * self.split_size, has_bias=False)
        self.I_fine = nn.Dense(3 * 4, 3 * self.split_size, has_bias=False)

        self.bias_u = mindspore.Parameter(initializer(Zero(), [self.hidden_size], mindspore.float32))
        self.bias_r = mindspore.Parameter(initializer(Zero(), [self.hidden_size], mindspore.float32))
        self.bias_e = mindspore.Parameter(initializer(Zero(), [self.hidden_size], mindspore.float32))

        self.num_params()

    def construct(self, prev_y, prev_hidden, current_coarse):
        """
        Args:
            prev_y: Tensor, 形状为 [B, 8]
            prev_hidden: Tensor, 形状为 [B, 384]
            current_coarse: Tensor, 形状为 [B, 4]
        Returns:
            c: Tensor, coarse 输出
            f: Tensor, fine 输出
            hidden: Tensor, 更新后的隐藏状态
        """
        R_hidden = self.R(prev_hidden)
        R_u, R_r, R_e = ops.Split(axis=1, output_num=3)(R_hidden)

        coarse_input_proj = self.I_coarse(prev_y)
        I_coarse_u, I_coarse_r, I_coarse_e = ops.Split(axis=1, output_num=3)(coarse_input_proj)

        fine_input = ops.Concat(axis=1)((prev_y, current_coarse))
        fine_input_proj = self.I_fine(fine_input)
        I_fine_u, I_fine_r, I_fine_e = ops.Split(axis=1, output_num=3)(fine_input_proj)

        I_u = ops.Concat(axis=1)((I_coarse_u, I_fine_u))
        I_r = ops.Concat(axis=1)((I_coarse_r, I_fine_r))
        I_e = ops.Concat(axis=1)((I_coarse_e, I_fine_e))

        u = ops.Sigmoid()(R_u + I_u + self.bias_u)
        r = ops.Sigmoid()(R_r + I_r + self.bias_r)
        e = ops.Tanh()(r * R_e + I_e + self.bias_e)
        hidden = u * prev_hidden + (1.0 - u) * e

        h_c, h_f = ops.Split(axis=1, output_num=2)(hidden)

        out_c = ops.ReLU()(self.O1(h_c))
        out_f = ops.ReLU()(self.O3(h_f))

        one_c = self.one_O2(out_c)
        one_f = self.one_O4(out_f)
        two_c = self.two_O2(out_c)
        two_f = self.two_O4(out_f)
        three_c = self.three_O2(out_c)
        three_f = self.three_O4(out_f)
        four_c = self.four_O2(out_c)
        four_f = self.four_O4(out_f)

        c = ops.Concat(axis=0)((one_c, two_c, three_c, four_c))
        f = ops.Concat(axis=0)((one_f, two_f, three_f, four_f))

        return c, f, hidden

    def get_initial_hidden(self, batch_size=1):
        hidden_state = mindspore.ops.Zeros()((batch_size, self.hidden_size), mindspore.float32)
        return hidden_state

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.get_parameters())
        parameters = sum([np.prod(p.shape) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = WaveRNN(hidden_size=384, quantization=256)
    model.set_train()

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

    batch_size = 16
    prev_y = Tensor(np.random.randn(batch_size, 8).astype(np.float32))
    prev_hidden = model.get_initial_hidden(batch_size)
    current_coarse = Tensor(np.random.randn(batch_size, 4).astype(np.float32))

    target_c = Tensor(np.random.randint(0, 256, (batch_size * 4,)), mindspore.int32)
    target_f = Tensor(np.random.randint(0, 256, (batch_size * 4,)), mindspore.int32)


    class LossNet(nn.Cell):
        def __init__(self, network):
            super(LossNet, self).__init__()
            self.network = network
            self.criterion = criterion

        def construct(self, prev_y, prev_hidden, current_coarse, target_c, target_f):
            c, f, hidden = self.network(prev_y, prev_hidden, current_coarse)
            loss_c = self.criterion(c, target_c)
            loss_f = self.criterion(f, target_f)
            loss = loss_c + loss_f
            return loss


    loss_net = LossNet(model)

    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    train_net.set_train()

    loss = train_net(prev_y, prev_hidden, current_coarse, target_c, target_f)
    print(f"Loss: {loss.asnumpy()}")