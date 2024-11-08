import math
import mindspore as ms
from mindspore import nn, ops
import numpy as np

from alibi.attention import ALiBiMultiHeadAttention
from alibi.config import ALiBiConfig

class FeedForward(nn.Cell):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        d_hidden = config.d_model * config.expansion_factor
        self.fc1 = nn.Dense(config.d_model, d_hidden)
        self.fc2 = nn.Dense(d_hidden, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def construct(self, x: ms.tensor) -> ms.tensor:
        x = ops.gelu(self.fc1(x))
        out = self.dropout(self.fc2(x))
        return out


class ALiBiTransformerLayer(nn.Cell):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.ffn_norm = nn.LayerNorm((config.d_model,))
        self.attn_norm = nn.LayerNorm((config.d_model,))
        self.ffn = FeedForward(config)
        self.attn = ALiBiMultiHeadAttention(config)

    def construct(self, x: ms.tensor) -> ms.tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x