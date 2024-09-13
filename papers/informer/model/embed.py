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
"""embed"""


import math

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import numpy as ms_np
from mindspore import dtype, Parameter
from mindspore.ops import ExpandDims

from utils.he_normal import HeNormal
from utils.trans_tools import trans_shape


class PositionalEmbedding(nn.Cell):
    """PositionalEmbedding"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = ms_np.zeros((max_len, d_model)).astype(dtype.float32)
        self.expand = ExpandDims()

        position = self.expand(ms_np.arange(0, max_len), 1).astype(dtype.float32)
        div_term = ms_np.exp(ms_np.arange(0, d_model, 2).astype(dtype.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = ms_np.sin(position * div_term)
        pe[:, 1::2] = ms_np.cos(position * div_term)

        pe = self.expand(pe, 0)
        self.register_buffer('pe', pe)

    def construct(self, x):
        return self.pe[:, :P.Shape()(x)[1]]


class TokenEmbedding(nn.Cell):
    """TokenEmbedding"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.token_conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, pad_mode='pad')
        self.henormal = HeNormal(mode='fan_in', nonlinearity='leaky_relu')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                self.henormal(m.weight.data)

    def construct(self, x):
        x = self.token_conv(P.Transpose()(x, (0, 2, 1,)))
        x = x.transpose(trans_shape((1, 2), x.shape))
        return x


class FixedEmbedding(nn.Cell):
    """TokenEmbedding"""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = ms_np.zeros((c_in, d_model)).astype(dtype.float32)
        w.require_grad = False
        self.expand = ExpandDims()

        position = self.expand(ms_np.arange(0, c_in), 1).astype(dtype.float32)
        div_term = ms_np.exp(ms_np.arange(0, d_model, 2).astype(dtype.float32) * -(math.log(10000.0) / d_model))

        w[:, 0::2] = ms_np.sin(position * div_term)
        w[:, 1::2] = ms_np.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = Parameter(w, requires_grad=False)

    def construct(self, x):
        return self.emb(x)


class TemporalEmbedding(nn.Cell):
    """TemporalEmbedding"""
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        self.embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = self.embed(4, d_model)
        self.hour_embed = self.embed(24, d_model)
        self.weekday_embed = self.embed(7, d_model)
        self.day_embed = self.embed(32, d_model)
        self.month_embed = self.embed(13, d_model)

    def construct(self, x):
        x = x.astype(dtype.float64)

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Cell):
    """TimeFeatureEmbedding"""
    def __init__(self, d_model, freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Dense(in_channels=d_inp, out_channels=d_model)

    def construct(self, x):
        return self.embed(x)


class DataEmbedding(nn.Cell):
    """DataEmbedding"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
                                  if embed_type != 'timeF' else \
                                  TimeFeatureEmbedding(d_model=d_model, freq=freq)

        self.dropout = nn.Dropout(keep_prob=1.0 - dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
