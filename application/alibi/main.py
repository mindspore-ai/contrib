import mindspore as ms
from mindspore import ops

from alibi import ALiBiConfig, ALiBiTransformer

def test_alibi_transformer():

    config = ALiBiConfig(
        d_model=256,           # 模型的维度
        num_heads=8,           # 注意力头的数量
        expansion_factor=4,    # 前馈网络的扩展因子
        dropout=0.1,           # Dropout 的概率
        causal=True,           # 是否使用因果掩码
        max_len=100            # 序列的最大长度
    )
    model = ALiBiTransformer(config)

    # shape :(batch_size, seq_len, d_model)
    batch_size, seq_len, d_model = 8, 100, 256
    x = ops.randn((batch_size, seq_len, d_model), seed=42) 

    output = model(ms.Tensor(x, ms.float32))

    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, but got {output.shape}"

    print("Test passed. Output shape:", output.shape)

if __name__ == '__main__':
    test_alibi_transformer()
