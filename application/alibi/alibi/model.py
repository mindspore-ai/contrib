import mindspore as ms
from mindspore import nn, ops

from alibi.config import ALiBiConfig
from alibi.layers import ALiBiTransformerLayer


class ALiBiTransformer(nn.Cell):
    def __init__(self, config: ALiBiConfig) -> None:
        super().__init__()
        self.max_len = config.max_len
        self.layers = nn.SequentialCell(
            *[ALiBiTransformerLayer(config) for _ in range(config.num_layers)]
        )

    def construct(self, x: ms.tensor) -> ms.tensor:
        _, seq_len, _ = x.shape
        assert seq_len <= self.max_len, "sequence length exceeds `max_len`"
        return self.layers(x)