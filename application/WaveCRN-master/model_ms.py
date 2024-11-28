import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class WaveCRN(nn.Cell):
    def __init__(self):
        super(WaveCRN, self).__init__()
        self.net = ConvBSRU(frame_size=96, conv_channels=256, stride=48, num_layers=6, dropout=0.0)

    def construct(self, x):
        return self.net(x)

class ConvBSRU(nn.Cell):
    def __init__(self, frame_size, conv_channels, stride=128, num_layers=1, dropout=0.1, rescale=False, bidirectional=True):
        super(ConvBSRU, self).__init__()
        num_directions = 2 if bidirectional else 1
        if stride == frame_size:
            padding = 0
        elif stride == frame_size // 2:
            padding = frame_size // 2
        else:
            print(stride, frame_size)
            raise ValueError(
                'Invalid stride {}. Length of stride must be "frame_size" or "0.5 * "frame_size"'.format(stride))
            
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=conv_channels, 
            kernel_size=frame_size, 
            stride=stride,
            has_bias=False
        )
        self.deconv = nn.Conv1dTranspose(
            in_channels=conv_channels,
            out_channels=1,
            kernel_size=frame_size,
            stride=stride,
            has_bias=False
        )
        self.outfc = nn.Dense(num_directions * conv_channels, conv_channels, has_bias=False)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=conv_channels,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def construct(self, x):
        output = self.conv(x) # B,C,D
        output_ = output.permute(2, 0, 1) # D, B, C
        output, _ = self.lstm(output_) # D, B, 2C
        output = self.outfc(output) # D, B, C
        #output = output_ * F.sigmoid(output)
        output = output_ * output # D, B, C
        output = output.permute(1, 2, 0) # B, C, D
        output = self.deconv(output)
        #output = self.conv11(output)
        output = ops.tanh(output)

        return output


import unittest
from unittest.mock import MagicMock
from mindspore import Tensor


class TestWaveCRN(unittest.TestCase):
    def test_wavecrn(self):
        # Instantiate the WaveCRN model
        model = WaveCRN()

        # Create a dummy input tensor
        input_tensor = ops.ones((1, 1, 96 * 10))

        # Run the model
        output = model(input_tensor)

        # Check the output shape
        expected_shape = (1, 1, 96 * 10)
        self.assertEqual(output.shape, expected_shape)

class TestConvBSRU(unittest.TestCase):
    def test_convbsru(self):
        # Instantiate the ConvBSRU model
        model = ConvBSRU(frame_size=96, conv_channels=256, stride=48, num_layers=6, dropout=0.0)

        # Create a dummy input tensor
        input_tensor = ops.ones((1, 1, 96 * 10))

        # Run the model
        output = model(input_tensor)

        # Check the output shape
        expected_shape = (1, 1, 96 * 10)
        self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()

