# Copyright 2022 Huawei Technologies Co., Ltd
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
"""TransformerEncoder module."""

import numpy as np

from mindspore import nn

from mindvision.classification.models.blocks import Attention, FeedForward, DropPath
from multi_model.Attention2 import MyAttention as Attention2


class MyLstm(nn.Cell):
    """
    lstm模块
    """

    def __init__(self, vocab_size, hidden_dim, embedding_dim, padding_idx=0):
        super(MyLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

    def construct(self, inputs, seq_length):
        embeds = self.embedding(inputs)
        outputs, _ = self.lstm(embeds, seq_length=seq_length)

        return outputs


class MytransformerDecoderLayer(nn.Cell):
    """
    TransformerEncoder implementation.

    Args:
        dim (int): The dimension of embedding.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        mlp_dim (int): The dimension of MLP hidden layer.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.LayerNorm.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = MytransformerDecoderLayer(768, 12, 12, 3072)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,

                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(MytransformerDecoderLayer, self).__init__()

        self.normalization1 = norm((dim,))
        self.normalization2 = norm((dim,))
        self.normalization3 = norm((dim,))
        self.attention = Attention(dim=dim,
                                   num_heads=num_heads,
                                   keep_prob=keep_prob,
                                   attention_keep_prob=attention_keep_prob)
        self.mul_attention = Attention2(dim=dim,
                                        num_heads=num_heads,
                                        keep_prob=keep_prob,
                                        attention_keep_prob=attention_keep_prob)

        self.feedforward = FeedForward(in_features=dim,
                                       hidden_features=mlp_dim,
                                       activation=activation,
                                       keep_prob=keep_prob)

    def construct(self, encode_input, decode_input):
        """Transformer construct."""

        encode_attention = self.attention(encode_input)

        encode_x = self.normalization1(encode_attention) + encode_input

        encode_decode_attention = self.mul_attention(encode_x, decode_input)
        enc_dec_normal = self.normalization2(encode_decode_attention) + encode_x

        fc = self.feedforward(enc_dec_normal)

        return self.normalization3(fc) + enc_dec_normal


class MytransformerDecoder(nn.Cell):
    """
    transformer模块
    """

    def __init__(self, dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 vocab_size: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,

                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):

        super(MytransformerDecoder, self).__init__()

        drop_path_rate = 1.0 - drop_path_keep_prob
        dpr = [i.item() for i in np.linspace(0, drop_path_rate, num_layers)]
        mlp_seeds = [np.random.randint(1024) for _ in range(num_layers)]

        layers = []
        for i in range(num_layers):

            if drop_path_rate > 0:
                layers.append(
                    nn.SequentialCell([MytransformerDecoderLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim,
                                                                 keep_prob=keep_prob,
                                                                 attention_keep_prob=attention_keep_prob,
                                                                 activation=activation,
                                                                 norm=norm),
                                       DropPath(dpr[i], mlp_seeds[i])
                                       ])
                )
            else:
                layers.append(
                    MytransformerDecoderLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim,
                                              keep_prob=keep_prob, attention_keep_prob=attention_keep_prob,
                                              activation=activation,
                                              norm=norm)

                )
        self.layers = nn.SequentialCell(layers)

        self.lstm = MyLstm(vocab_size=vocab_size, hidden_dim=dim, embedding_dim=100)

    def construct(self, encode_input, decode_input, seq_length):
        decode_input = self.lstm(decode_input, seq_length)
        for layer in self.layers:
            decode_out = layer(encode_input, decode_input)
            decode_input = decode_out

        return decode_out


if __name__ == '__main__':
    model = MytransformerDecoder(dim=768, num_layers=5, num_heads=8, mlp_dim=128, vocab_size=100)
    print(model.trainable_params())
