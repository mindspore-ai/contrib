#============= model_builder.py =============
from mindspore import nn
from mindspore.common.initializer import XavierUniform, Normal

class MortalityPredictor(nn.Cell):
    """死亡率预测模型（兼容CPU/GPU）"""
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Dense(
            in_channels=input_size,
            out_channels=1,
            weight_init=XavierUniform(),
            bias_init=Normal(0.02),
            activation='sigmoid'
        )
    
    def construct(self, x):
        return self.fc(x)

def create_model(input_dim: int) -> nn.Cell:
    """创建并初始化模型"""
    return MortalityPredictor(input_dim)