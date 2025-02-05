import math
import mindspore.nn as nn
from mindspore.common.initializer import Normal, Zero, initializer

class FSRCNN(nn.Cell):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        # 第一部分：特征提取
        self.first_part = nn.SequentialCell([
            nn.Conv2d(num_channels, d, kernel_size=5, pad_mode='pad', padding=5//2, has_bias=True),
            nn.PReLU()
        ])
        
        # 中间部分：非线性映射
        self.mid_part = nn.SequentialCell()
        # 收缩层
        self.mid_part.append(nn.Conv2d(d, s, kernel_size=1, pad_mode='pad', padding=0, has_bias=True))
        self.mid_part.append(nn.PReLU())
        # 多个映射层
        for _ in range(m):
            self.mid_part.append(nn.Conv2d(s, s, kernel_size=3, pad_mode='pad', padding=3//2, has_bias=True))
            self.mid_part.append(nn.PReLU())
        # 扩展层
        self.mid_part.append(nn.Conv2d(s, d, kernel_size=1, pad_mode='pad', padding=0, has_bias=True))
        self.mid_part.append(nn.PReLU())
        
        # 最后部分：反卷积上采样
        self.last_part = nn.Conv2dTranspose(
            d, num_channels, kernel_size=9, 
            stride=scale_factor, 
            pad_mode='pad', 
            padding=9//2,
            output_padding=scale_factor-1,
            has_bias=True
        )
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """自定义权重初始化方法"""
        for name, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
                # 计算标准差
                fan_in = cell.in_channels * cell.kernel_size[0] * cell.kernel_size[1]
                std = math.sqrt(2.0 / fan_in)
                if isinstance(cell, nn.Conv2dTranspose):
                    std = 0.001  # 最后一层使用更小的标准差
                
                # 初始化权重
                cell.weight.set_data(
                    initializer(Normal(std), cell.weight.shape)
                )
                
                # 初始化偏置
                if cell.has_bias:
                    cell.bias.set_data(initializer(Zero(), cell.bias.shape))

    def construct(self, x):
        """前向传播"""
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

def print_model_info(scale_factor=4):
    """打印模型信息"""
    # 创建模型实例
    model = FSRCNN(scale_factor=scale_factor)
    
    # 打印模型结构
    print("="*60)
    print("模型结构:")
    for name, cell in model.cells_and_names():
        if not cell._cells:  # 仅显示叶子节点
            print(f"{name}: {cell}")
    
    # 计算并打印参数总数
    total_params = sum(p.data.size for p in model.get_parameters())
    print("\n" + "="*60)
    print(f"总参数量 (scale_factor={scale_factor}): {total_params:,}")
    print("="*60)

if __name__ == "__main__":
    print_model_info(scale_factor=4)