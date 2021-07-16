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
"""decoder"""

import mindspore.nn as nn

from utils.trans_tools import trans_shape


class DecoderLayer(nn.Cell):
    """DecoderLayer"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, epsilon=1e-05)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, epsilon=1e-05)
        self.norm3 = nn.LayerNorm(normalized_shape=d_model, epsilon=1e-05)
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout)
        self.activation = nn.get_activation(activation)

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        """build"""
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(trans_shape((-1, 1), y.shape)))))
        y = self.dropout(self.conv2(y).transpose(trans_shape((-1, 1), self.conv2(y).shape)))

        return self.norm3(x + y)


class Decoder(nn.Cell):
    """Decoder"""
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
