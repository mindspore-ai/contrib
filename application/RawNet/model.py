#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_channels=in_planes, 
                     out_channels=out_planes, 
                     kernel_size=3, 
                     stride=stride, 
                     pad_mode="pad", 
                     padding=1, 
                     has_bias=False)


class BasicBlock3x3(nn.Cell):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(alpha=0.01)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.add = ops.Add()
        self.stride = stride

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu(out)

        return out


class RawNet(nn.Cell):
    def __init__(self, input_channel, num_classes=1211):
        super(RawNet, self).__init__()
        self.inplanes3 = 128

        # First convolutional layer
        self.conv1 = nn.Conv1d(input_channel, 128, kernel_size=3, stride=3, pad_mode="valid", has_bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU(alpha=0.01)

        # ResBlocks and MaxPooling layers
        self.resblock_1_1 = self._make_layer3(BasicBlock3x3, 128, 1, stride=1)
        self.resblock_1_2 = self._make_layer3(BasicBlock3x3, 128, 1, stride=1)
        self.maxpool_resblock_1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.resblock_2_1 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_2 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_3 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_4 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.maxpool_resblock_2 = nn.MaxPool1d(kernel_size=3, stride=3)

        # GRU layer
        self.gru = nn.GRU(input_size=256, hidden_size=1024, batch_first=True, dropout=0.2, bidirectional=False)

        # Fully connected layers
        self.spk_emb = nn.Dense(1024, 128)
        self.output_layer = nn.Dense(128, num_classes)


    def _make_layer3(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv1d(self.inplanes3, planes * block.expansion, kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            ])

        layers = [block(self.inplanes3, planes, stride, downsample)]
        self.inplanes3 = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.SequentialCell(layers)


    def construct(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        # ResBlock-1
        out = self.resblock_1_1(out)
        out = self.maxpool_resblock_1(out)
        out = self.resblock_1_2(out)
        out = self.maxpool_resblock_1(out)

        # ResBlock-2
        out = self.resblock_2_1(out)
        out = self.maxpool_resblock_2(out)
        out = self.resblock_2_2(out)
        out = self.maxpool_resblock_2(out)
        out = self.resblock_2_3(out)
        out = self.maxpool_resblock_2(out)
        out = self.resblock_2_4(out)
        out = self.maxpool_resblock_2(out)

        # GRU
        out = ops.Transpose()(out, (0, 2, 1))
        out, _ = self.gru(out)
        out = ops.Transpose()(out, (0, 2, 1))

        spk_embeddings = self.spk_emb(out[:, :, -1])
        preds = self.output_layer(spk_embeddings)

        return preds, spk_embeddings


def main():
    
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  # "CPU"/"GPU"/"Ascend"

    batch_size = 8
    input_channels = 1
    input_length = 16000
    num_classes = 1211

    inputs = Tensor(np.random.randn(batch_size, input_channels, input_length).astype(np.float32))

    model = RawNet(input_channel=input_channels, num_classes=num_classes)

    preds, spk_embeddings = model(inputs)

    print("Predictions shape:", preds.shape)
    print("Speaker embeddings shape:", spk_embeddings.shape)


if __name__ == "__main__":
    main()