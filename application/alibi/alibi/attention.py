import math
import mindspore as ms
from mindspore import nn, ops, Parameter
import numpy as np

from alibi.config import ALiBiConfig


def get_relative_positions(seq_len):
    x = ops.arange(seq_len)[None, :]
    y = ops.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        ms.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class ALiBiMultiHeadAttention(nn.Cell):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.causal = config.causal
        self.num_heads = config.num_heads
        self.scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        # self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.m = Parameter(ms.Tensor(get_alibi_slope(self.num_heads)), requires_grad=False)
        
        self.kqv = nn.Dense(config.d_model, 3 * config.d_model, has_bias=False)
        if config.causal:
            # self.register_buffer(
            #     "mask", ops.tril(ops.ones((1, 1, config.max_len, config.max_len)))
            # )
            mask = np.tril(np.ones((1, 1, config.max_len, config.max_len), dtype=np.float32))
            self.mask = Parameter(ms.Tensor(mask), requires_grad=False)

    def construct(self, x):
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, axis=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).swapaxes(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).swapaxes(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)

        bias = (self.m * get_relative_positions(seq_len)).unsqueeze(0)
        # bias.shape == (1, num_heads, seq_len, seq_len)

        score = ops.matmul(query, key) / self.scale + bias
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        attn = ops.softmax(score, axis=-1)
        out = ops.matmul(attn, value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.swapaxes(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.dropout(out)

        return out