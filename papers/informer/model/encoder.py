# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""encoder"""

import mindspore.nn as nn
import mindspore.ops.operations as P

from utils.trans_tools import trans_shape


class ConvLayer(nn.Cell):
    """ConvLayer"""
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.down_conv = nn.Conv1d(in_channels=c_in,
                                   out_channels=c_in,
                                   kernel_size=3,
                                   padding=padding,
                                   pad_mode='pad')
        self.norm = nn.BatchNorm1d(num_features=c_in, momentum=0.9)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.pad = self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), "CONSTANT")

    def construct(self, x):
        x = self.down_conv(P.Transpose()(x, (0, 2, 1,)))
        x = self.norm(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.maxpool(x)
        x = x.transpose(trans_shape((1, 2), x.shape))
        return x


class EncoderLayer(nn.Cell):
    """EncoderLayer"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, epsilon=1e-05)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, epsilon=1e-05)
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def construct(self, x, attn_mask=None):
        """build"""
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(trans_shape((-1, 1), y.shape)))))
        y = self.dropout(self.conv2(y).transpose(trans_shape((-1, 1), self.conv2(y).shape)))

        return self.norm2(x + y), attn


class Encoder(nn.Cell):
    """Encoder"""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        """build"""
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Cell):
    """EncoderStack"""
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.CellList(encoders)
        self.inp_lens = inp_lens

    def construct(self, x):
        """build"""
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = P.Concat(-2)(x_stack)

        return x_stack, attns
