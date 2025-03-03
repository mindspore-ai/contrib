import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Normal, initializer
import numpy as np


class Tisa(nn.Cell):
    """TISA模型的MindSpore实现，用于计算自注意力中的位置贡献矩阵"""
    
    def __init__(self, num_attention_heads: int = 12, num_kernels: int = 5):
        """
        初始化TISA模型
        
        参数:
            num_attention_heads (int): 注意力头的数量，默认值为12
            num_kernels (int): 核的数量，默认值为5
        """
        super().__init__()
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.num_kernels = num_kernels  # 核的数量

        # 定义可学习的参数，使用Normal初始化器
        self.kernel_offsets = Parameter(
            initializer(Normal(sigma=5.0), (self.num_kernels, self.num_attention_heads)),
            name="kernel_offsets"
        )  # 核偏移量
        self.kernel_amplitudes = Parameter(
            initializer(Normal(mean=0.1, sigma=0.01), (self.num_kernels, self.num_attention_heads)),
            name="kernel_amplitudes"
        )  # 核幅度
        self.kernel_sharpness = Parameter(
            initializer(Normal(mean=0.1, sigma=0.01), (self.num_kernels, self.num_attention_heads)),
            name="kernel_sharpness"
        )  # 核锐度

    def create_relative_offsets(self, seq_len: int):
        """创建从 -seq_len + 1 到 seq_len - 1 的所有相对距离偏移量
        
        参数:
            seq_len (int): 序列长度
            
        返回:
            Tensor: 相对距离偏移量张量
        """
        return ops.arange(-seq_len, seq_len + 1, dtype=ms.float32)

    def compute_positional_scores(self, relative_offsets):
        """根据序列长度计算每个相对距离的位置得分，使用径向基函数（RBF）
        
        参数:
            relative_offsets (Tensor): 相对距离偏移量
            
        返回:
            Tensor: 位置得分张量
        """
        offsets_exp = self.kernel_offsets.expand_dims(-1)  
        sharpness_exp = self.kernel_sharpness.expand_dims(-1)  
        rbf_scores = (
            self.kernel_amplitudes.expand_dims(-1) *  # 扩展维度以广播
            ops.exp(
                -ops.abs(sharpness_exp) *
                ((offsets_exp - relative_offsets) ** 2)  # 偏移差的平方
            )
        )
        return rbf_scores.sum(axis=0)  # 按核维度求和

    def scores_to_toeplitz_matrix(self, positional_scores, seq_len: int):
        """将TISA位置得分转换为自注意力方程的最终矩阵
        
        参数:
            positional_scores (Tensor): 位置得分
            seq_len (int): 序列长度
            
        返回:
            Tensor: Toeplitz矩阵
        """
        base_indices = ops.arange(seq_len).reshape(-1, 1)
        deformed_toeplitz = (
            (ops.arange(0, -(seq_len ** 2), -1, dtype=ms.int32) + (seq_len - 1))
            .reshape(seq_len, seq_len) +
            (seq_len + 1) * base_indices
        )
        expanded_scores = ops.gather(
            positional_scores,
            deformed_toeplitz.reshape(-1).astype(ms.int32),
            1
        )
        return expanded_scores.reshape(self.num_attention_heads, seq_len, seq_len)

    def construct(self, seq_len: int):
        """计算自注意力模块中平移不变的位置贡献矩阵
        
        参数:
            seq_len (int): 序列长度
            
        返回:
            Tensor: 位置贡献矩阵
        """
        if self.num_kernels == 0:
            return ops.zeros((self.num_attention_heads, seq_len, seq_len), ms.float32)
        offsets = self.create_relative_offsets(seq_len)
        pos_scores = self.compute_positional_scores(offsets)
        return self.scores_to_toeplitz_matrix(pos_scores, seq_len)

    def visualize(self, seq_len: int = 10, attention_heads=None, save_path=None):
        """Visualize TISA positional scores, plotting results for each attention head with relative distance
        
        Parameters:
            seq_len (int): Sequence length, default is 10
            attention_heads (list, optional): Indices of attention heads to visualize, default is all
            save_path (str, optional): Path to save the image, if None, display the image
        """
        if attention_heads is None:
            attention_heads = list(range(self.num_attention_heads))
        x = self.create_relative_offsets(seq_len).asnumpy()
        y = (
            self.compute_positional_scores(self.create_relative_offsets(seq_len))
            .asnumpy()
        )
        plt.figure(figsize=(10, 6))
        for i in attention_heads:
            plt.plot(x, y[i])
        plt.xlim(x.min(), x.max())
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        plt.close()


def set_tisa_parameters_from_files(tisa, kernel_offsets_path, kernel_amplitudes_path, kernel_sharpness_path):
    """从文件中加载参数值并设置到 Tisa 模型中
    
    参数:
        tisa: Tisa 模型实例
        kernel_offsets_path (str): kernel_offsets 文件路径
        kernel_amplitudes_path (str): kernel_amplitudes 文件路径
        kernel_sharpness_path (str): kernel_sharpness 文件路径
    """
    kernel_offsets_data = np.loadtxt(kernel_offsets_path).reshape((5, 12))
    kernel_amplitudes_data = np.loadtxt(kernel_amplitudes_path).reshape((5, 12))
    kernel_sharpness_data = np.loadtxt(kernel_sharpness_path).reshape((5, 12))

    
    tisa.kernel_offsets.set_data(ms.Tensor(kernel_offsets_data, ms.float32))
    tisa.kernel_amplitudes.set_data(ms.Tensor(kernel_amplitudes_data, ms.float32))
    tisa.kernel_sharpness.set_data(ms.Tensor(kernel_sharpness_data, ms.float32))


def run_tisa_visualization_fixed(seq_len=20, save_path='tisa_visualization_fixed.png',
                                kernel_offsets_path='kernel_offsets.txt',
                                kernel_amplitudes_path='kernel_amplitudes.txt',
                                kernel_sharpness_path='kernel_sharpness.txt'):
    """使用固定参数运行 TISA 模型并保存可视化图像，用于与 PyTorch 版本一致
    
    参数:
        seq_len (int): 序列长度，默认值为20
        save_path (str): 保存可视化图像的路径，默认值为'tisa_visualization_fixed.png'
        kernel_offsets_path (str): kernel_offsets 文件路径
        kernel_amplitudes_path (str): kernel_amplitudes 文件路径
        kernel_sharpness_path (str): kernel_sharpness 文件路径
    """
    tisa = Tisa()  
    set_tisa_parameters_from_files(tisa, kernel_offsets_path, kernel_amplitudes_path, kernel_sharpness_path)
    tisa(seq_len) 
    tisa.visualize(seq_len=seq_len, save_path=save_path)  


def run_tisa_visualization_random(seq_len=20, save_path=None):
    """使用随机种子运行 TISA 模型并显示可视化图像，保留原始随机生成效果
    
    参数:
        seq_len (int): 序列长度，默认值为20
        save_path (str, optional): 如果提供，则保存图像；否则显示图像，默认值为None
    """
    tisa = Tisa()  
    tisa(seq_len)  
    tisa.visualize(seq_len=seq_len, save_path=save_path) 


if __name__ == "__main__":

    # 使用固定种子生成并保存图像和参数 以便对照pytorch版本验证是否成功复刻。
    print("运行固定参数版本...")
    run_tisa_visualization_fixed()

    # 原代码主函数：使用随机种子生成并显示图像
    print("运行随机种子版本...")
    run_tisa_visualization_random()