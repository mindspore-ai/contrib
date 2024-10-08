from mindspore import nn, ops
import mindspore


class ParameterSharedTransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        d_model=512,
        nhead=16,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_unique_layers=3,
        num_total_layers=6,
        mode="cycle_rev",
        norm=False,
    ):
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0
        if mode == "cycle_rev":
            assert quotient == 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        super().__init__(encoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if norm else None

    def construct(self, x, mask=None, src_key_padding_mask=None, verbose=False):
        for i in range(self.N):
            if self.mode == "sequence":
                i = i // (self.N // self.M)
            elif self.mode == "cycle":
                i = i % self.M
            elif i > (self.N - 1) / 2:
                i = self.N - i - 1
            if verbose:
                print(f"layer {i}")
            x = self.layers[i](x, mask, src_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


if __name__ == "__main__":
    x = ops.randn(8, 100, 512)  # (batch_size, seq_len, d_model)
    model = ParameterSharedTransformerEncoder()
    print("Cycle Reverse Mode:")
    print(model(x, verbose=True).shape)
    print("Cycle Mode:")
    model = ParameterSharedTransformerEncoder(mode="cycle")
    print(model(x, verbose=True).shape)
    print("Sequence Mode:")
    model = ParameterSharedTransformerEncoder(mode="sequence")
    print(model(x, verbose=True).shape)
