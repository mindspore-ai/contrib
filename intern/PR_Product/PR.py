import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import float32


class PRLinear(nn.Dense):

    def __init__(self, in_features, out_features, has_bias=True, eps=1e-8):
        super(PRLinear, self).__init__(in_features, out_features, has_bias=has_bias)
        self.eps = eps

    def construct(self, x):
        # compute the length of w and x. We find this is faster than the norm, although the later is simple.
        w_len = ops.sqrt((ops.t(self.weight.pow(2).sum(axis=1, keepdims=True))).clamp(min=self.eps))  # 1*num_classes
        x_len = ops.sqrt((x.pow(2).sum(axis=1, keepdims=True)).clamp(min=self.eps))  # batch*1

        # compute the cosine of theta and abs(sine) of theta.
        wx_len = ops.matmul(x_len, w_len).clamp(min=self.eps)
        cos_theta = (ops.matmul(x, ops.t(self.weight)) / wx_len).clamp(-1.0, 1.0)  # batch*num_classes
        abs_sin_theta = ops.sqrt(1.0 - cos_theta ** 2)  # batch*num_classes

        # PR Product
        out = wx_len * (ops.stop_gradient(abs_sin_theta) * cos_theta + ops.stop_gradient(cos_theta) * (1.0 - abs_sin_theta))

        # to save memory
        del w_len, x_len, wx_len, cos_theta, abs_sin_theta

        if self.has_bias is not None:
            out = out + self.bias

        return out


class PRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 pad_mode='same',padding=0, dilation=1, group=1, has_bias=True, eps=1e-8):
        super(PRConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,pad_mode, padding, dilation, group, has_bias=has_bias)

        assert group == 1, 'Currently, we do not realize the PR for group CNN. Maybe you can do it yourself and welcome for pull-request.'

        self.eps = eps
        self.ones_weight = ops.ones((1, 1, self.weight.shape[2], self.weight.shape[3])).astype(float32)

    def construct(self, input):
        # compute the length of w
        w_len = ops.sqrt((self.weight.view(self.weight.shape[0], -1).pow(2).sum(axis=1, keepdims=True).t()).clamp(
            min=self.eps))  # 1*out_channels

        # compute the length of x at each position with the help of convolutional operation
        x_len = input.pow(2).sum(axis=1, keepdims=True)  # batch*1*H_in*W_in
        x_len = ops.sqrt((ops.conv2d(x_len, self.ones_weight,
                                     stride=self.stride,
                                      padding=self.padding, dilation=self.dilation,  groups=self.group)).clamp(
            min=self.eps))  # batch*1*H_out*W_out

        # compute the cosine of theta and abs(sine) of theta.
        wx_len = (x_len * (w_len.unsqueeze(-1).unsqueeze(-1))).clamp(min=self.eps)  # batch*out_channels*H_out*W_out
        cos_theta = (ops.conv2d(input, self.weight, None, self.stride,
                              padding=self.padding, dilation=self.dilation, groups=self.group) / wx_len).clamp(-1.0,
                                                                                         1.0)  # batch*out_channels*H_out*W_out
        abs_sin_theta = ops.sqrt(1.0 - cos_theta ** 2)

        # PR Product
        out = wx_len * (ops.stop_gradient(abs_sin_theta) * cos_theta + ops.stop_gradient(cos_theta) * (1.0 - abs_sin_theta))
        # to save memory
        del w_len, x_len, wx_len, cos_theta, abs_sin_theta

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class PRLSTMCell(nn.Cell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(PRLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # replace linear with PRLinear
        self.ih_linear = PRLinear(input_size, 4 * hidden_size, bias)
        self.hh_linear = PRLinear(hidden_size, 4 * hidden_size, bias)

        self.weight_ih = self.ih_linear.weight
        self.bias_ih = self.ih_linear.bias

        self.weight_hh = self.hh_linear.weight
        self.bias_hh = self.hh_linear.bias

    def construct(self, input, hidden):
        hx, cx = hidden

        gates = self.ih_linear(input) + self.hh_linear(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = ops.sigmoid(ingate)
        forgetgate = ops.sigmoid(forgetgate)
        cellgate = ops.tanh(cellgate)
        outgate = ops.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * ops.tanh(cy)

        return hy, cy


# 测试 PRLinear 类
def test_PRLinear():
    in_features = 5
    out_features = 3
    batch_size = 2

    x = ops.randn(batch_size, in_features)
    pr_linear = PRLinear(in_features, out_features)

    output = pr_linear(x)
    print("PRLinear output:", output)
    
# 测试 PRConv2d 类
def test_PRConv2d():
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    batch_size = 1
    height, width = 5, 5

    x = ops.randn(batch_size, in_channels, height, width)
    pr_conv2d = PRConv2d(in_channels, out_channels, kernel_size,pad_mode="valid")

    output = pr_conv2d(x)
    print("PRConv2d output:", output)
    
# # 测试 PRLSTMCell 类
def test_PRLSTMCell():
    input_size = 4
    hidden_size = 3
    batch_size = 1

    x = ops.randn(batch_size, input_size)
    hx = ops.randn(batch_size, hidden_size)
    cx = ops.randn(batch_size, hidden_size)
    hidden = (hx, cx)

    pr_lstm_cell = PRLSTMCell(input_size, hidden_size)

    hy, cy = pr_lstm_cell(x, hidden)
    print("PRLSTMCell hy:", hy)
    print("PRLSTMCell cy:", cy)
    

if __name__ == "__main__":
    test_PRLinear()
    test_PRConv2d()
    test_PRLSTMCell()