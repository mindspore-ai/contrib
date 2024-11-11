import mindspore
from mindspore import nn, Tensor, ops, context
import numpy as np

EPS = 1e-8

def conv_insn_lrelu(in_channel, out_channel, kernel_size_in, stride_in, padding_in, insn=True, lrelu=True):
    layers = []

    if isinstance(padding_in, int):
        padding = (padding_in, padding_in, padding_in, padding_in)
    elif isinstance(padding_in, tuple) and len(padding_in) == 2:
        padding = (padding_in[0], padding_in[0], padding_in[1], padding_in[1])
    elif isinstance(padding_in, tuple) and len(padding_in) == 4:
        padding = padding_in
    else:
        raise ValueError('Invalid padding_in')

    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size_in, stride=stride_in,
                            pad_mode='pad', padding=padding, has_bias=True))
    if insn:
        layers.append(nn.GroupNorm(num_groups=out_channel, num_channels=out_channel, affine=False))
    if lrelu:
        layers.append(nn.LeakyReLU(alpha=0.2))
    return nn.SequentialCell(layers)

def convt_insn_lrelu(in_channel, out_channel, kernel_size_in, stride_in, padding_in, insn=True, lrelu=True):
    layers = []

    if isinstance(padding_in, int):
        padding = (padding_in, padding_in, padding_in, padding_in)
    elif isinstance(padding_in, tuple) and len(padding_in) == 2:
        padding = (padding_in[0], padding_in[0], padding_in[1], padding_in[1])
    elif isinstance(padding_in, tuple) and len(padding_in) == 4:
        padding = padding_in
    else:
        raise ValueError('Invalid padding_in')

    layers.append(nn.Conv2dTranspose(in_channel, out_channel, kernel_size=kernel_size_in, stride=stride_in,
                                     pad_mode='pad', padding=padding, has_bias=True))
    if insn:
        layers.append(nn.GroupNorm(num_groups=out_channel, num_channels=out_channel, affine=False))
    if lrelu:
        layers.append(nn.LeakyReLU(alpha=0.2))
    return nn.SequentialCell(layers)

class EDNet_uncertainty(nn.Cell):
    def __init__(self, input_channel=1):
        super(EDNet_uncertainty, self).__init__()

        self.conv1 = conv_insn_lrelu(input_channel, 16, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.conv2 = conv_insn_lrelu(16, 32, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.conv3 = conv_insn_lrelu(32, 64, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.conv4 = conv_insn_lrelu(64, 128, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.conv5 = conv_insn_lrelu(128, 256, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.conv6 = conv_insn_lrelu(256, 512, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))

        self.convt6 = convt_insn_lrelu(512, 256, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.convt5 = convt_insn_lrelu(512, 128, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.convt4 = convt_insn_lrelu(256, 64, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.convt3 = convt_insn_lrelu(128, 32, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.convt2 = convt_insn_lrelu(64, 16, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))
        self.convt1 = convt_insn_lrelu(32, 16, kernel_size_in=(5, 5), stride_in=(1, 2), padding_in=(2, 2))

        self.convt1_mean = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1,
                                           padding_in=0, insn=False, lrelu=False)
        self.convt1_logvar = conv_insn_lrelu(16, out_channel=1, kernel_size_in=1, stride_in=1,
                                             padding_in=0, insn=False, lrelu=False)

        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(axis=1)
        self.sigmoid = ops.Sigmoid()
        self.concat = ops.Concat(axis=1)
        self.exp = ops.Exp()
        self.sqrt = ops.Sqrt()
        self.pow = ops.Pow()
        self.div = ops.Div()

    def construct(self, x, noisy_complex_real, noisy_complex_imag):
        x_expanded = self.expand_dims(x, 1)
        conv1 = self.conv1(x_expanded)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        convt6 = self.convt6(conv6)
        y = self.concat((convt6, conv5))

        convt5 = self.convt5(y)
        y = self.concat((convt5, conv4))

        convt4 = self.convt4(y)
        y = self.concat((convt4, conv3))

        convt3 = self.convt3(y)
        y = self.concat((convt3, conv2))

        convt2 = self.convt2(y)
        y = self.concat((convt2, conv1))

        convt1 = self.convt1(y)

        mean = self.sigmoid(self.convt1_mean(convt1))
        mean_squeezed = self.squeeze(mean)
        logvar = self.squeeze(self.convt1_logvar(convt1))

        WF_stft_real = mean_squeezed * noisy_complex_real
        WF_stft_imag = mean_squeezed * noisy_complex_imag

        x_squeezed = self.squeeze(x_expanded)
        denominator = 4 * self.pow(x_squeezed, 2) + EPS
        element = self.pow(0.5 * mean_squeezed, 2) + self.div(self.exp(logvar), denominator)
        approximated_map = 0.5 * mean_squeezed + self.sqrt(element + EPS)
        AMAP_stft_real = approximated_map * noisy_complex_real
        AMAP_stft_imag = approximated_map * noisy_complex_imag

        return (WF_stft_real, WF_stft_imag), (AMAP_stft_real, AMAP_stft_imag), logvar

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    input_data = np.random.randn(16, 150, 257).astype(np.float32)
    input_tensor = Tensor(input_data)

    noisy_real = np.random.randn(16, 150, 257).astype(np.float32)
    noisy_imag = np.random.randn(16, 150, 257).astype(np.float32)
    noisy_complex_real = Tensor(noisy_real)
    noisy_complex_imag = Tensor(noisy_imag)

    model = EDNet_uncertainty()

    (WF_stft_real, WF_stft_imag), (AMAP_stft_real, AMAP_stft_imag), logvar = model(
        input_tensor, noisy_complex_real, noisy_complex_imag
    )

    print("WF_stft_real shape:", WF_stft_real.shape)
    print("WF_stft_imag shape:", WF_stft_imag.shape)
    print("AMAP_stft_real shape:", AMAP_stft_real.shape)
    print("AMAP_stft_imag shape:", AMAP_stft_imag.shape)
    print("logvar shape:", logvar.shape)