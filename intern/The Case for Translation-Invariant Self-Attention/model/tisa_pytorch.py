import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

class Tisa(nn.Module):
    """TISA模型的PyTorch实现，用于计算自注意力中的位置贡献矩阵"""
    
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

        # 定义可学习的参数
        self.kernel_offsets = nn.Parameter(
            torch.Tensor(self.num_kernels, self.num_attention_heads)
        )  # 核偏移量
        self.kernel_amplitudes = nn.Parameter(
            torch.Tensor(self.num_kernels, self.num_attention_heads)
        )  # 核幅度
        self.kernel_sharpness = nn.Parameter(
            torch.Tensor(self.num_kernels, self.num_attention_heads)
        )  # 核锐度
        self._init_weights()  # 初始化权重

    def create_relative_offsets(self, seq_len: int):
        """创建从 -seq_len + 1 到 seq_len - 1 的所有相对距离偏移量"""
        return torch.arange(-seq_len, seq_len + 1)

    def compute_positional_scores(self, relative_offsets):
        """根据序列长度计算每个相对距离的位置得分，使用径向基函数（RBF）"""
        rbf_scores = (
            self.kernel_amplitudes.unsqueeze(-1)  # 扩展维度以广播
            * torch.exp(
                -torch.abs(self.kernel_sharpness.unsqueeze(-1))  # 锐度取绝对值
                * ((self.kernel_offsets.unsqueeze(-1) - relative_offsets) ** 2)  # 偏移差的平方
            )
        ).sum(axis=0)  # 按核维度求和
        return rbf_scores

    def scores_to_toeplitz_matrix(self, positional_scores, seq_len: int):
        """将TISA位置得分转换为自注意力方程的最终矩阵"""
        deformed_toeplitz = (
            (
                (torch.arange(0, -(seq_len ** 2), step=-1) + (seq_len - 1)).view(
                    seq_len, seq_len
                )
                + (seq_len + 1) * torch.arange(seq_len).view(-1, 1)
            )
            .view(-1)
            .long()
            .to(device=positional_scores.device)
        )
        expanded_positional_scores = torch.take_along_dim(
            positional_scores, deformed_toeplitz.view(1, -1), 1
        ).view(self.num_attention_heads, seq_len, seq_len)
        return expanded_positional_scores

    def forward(self, seq_len: int):
        """计算自注意力模块中平移不变的位置贡献矩阵"""
        if not self.num_kernels:
            return torch.zeros((self.num_attention_heads, seq_len, seq_len))
        positional_scores_vector = self.compute_positional_scores(
            self.create_relative_offsets(seq_len)
        )
        positional_scores_matrix = self.scores_to_toeplitz_matrix(
            positional_scores_vector, seq_len
        )
        return positional_scores_matrix

    def visualize(self, seq_len: int = 10, attention_heads=None, save_path=None):
        """可视化TISA的位置得分，随相对距离变化绘制每个注意力头的结果"""
        if attention_heads is None:
            attention_heads = list(range(self.num_attention_heads))
        x = self.create_relative_offsets(seq_len).detach().numpy()
        y = (
            self.compute_positional_scores(self.create_relative_offsets(seq_len))
            .detach()
            .numpy()
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

    def _init_weights(self):
        """初始化权重"""
        ampl_init_mean = 0.1 
        sharpness_init_mean = 0.1 
        torch.nn.init.normal_(self.kernel_offsets, mean=0.0, std=5.0)
        torch.nn.init.normal_(
            self.kernel_amplitudes, mean=ampl_init_mean, std=0.1 * ampl_init_mean
        )
        torch.nn.init.normal_(
            self.kernel_sharpness,
            mean=sharpness_init_mean,
            std=0.1 * sharpness_init_mean,
        )

def save_tisa_parameters(tisa, kernel_offsets_path, kernel_amplitudes_path, kernel_sharpness_path):
    """保存 Tisa 模型的参数值到文件
    
    参数:
        tisa: Tisa 模型实例
        kernel_offsets_path (str): 保存 kernel_offsets 的文件路径
        kernel_amplitudes_path (str): 保存 kernel_amplitudes 的文件路径
        kernel_sharpness_path (str): 保存 kernel_sharpness 的文件路径
    """
    np.savetxt(kernel_offsets_path, tisa.kernel_offsets.detach().numpy(), fmt='%.8f')
    np.savetxt(kernel_amplitudes_path, tisa.kernel_amplitudes.detach().numpy(), fmt='%.8f')
    np.savetxt(kernel_sharpness_path, tisa.kernel_sharpness.detach().numpy(), fmt='%.8f')
    print(f"Parameters saved: {kernel_offsets_path}, {kernel_amplitudes_path}, {kernel_sharpness_path}")

def run_tisa_visualization_fixed(seed=42, seq_len=20, save_path='tisa_visualization_fixed.png',
                                kernel_offsets_path='kernel_offsets.txt',
                                kernel_amplitudes_path='kernel_amplitudes.txt',
                                kernel_sharpness_path='kernel_sharpness.txt'):
    """
    使用固定随机种子运行TISA模型并保存可视化图像和参数值
    
    参数:
        seed (int): 随机种子，默认值为42
        seq_len (int): 序列长度，默认值为20
        save_path (str): 保存可视化图像的路径，默认值为'tisa_visualization_fixed.png'
        kernel_offsets_path (str): 保存 kernel_offsets 的文件路径
        kernel_amplitudes_path (str): 保存 kernel_amplitudes 的文件路径
        kernel_sharpness_path (str): 保存 kernel_sharpness 的文件路径
    """
    torch.manual_seed(seed)  
    tisa = Tisa()  
    tisa(seq_len)  
    tisa.visualize(seq_len=seq_len, save_path=save_path) 
    save_tisa_parameters(tisa, kernel_offsets_path, kernel_amplitudes_path, kernel_sharpness_path)  # 保存参数值

def run_tisa_visualization_random(seq_len=20, save_path=None):
    """
    使用随机种子运行TISA模型并显示可视化图像，保留原始随机生成效果
    
    参数:
        seq_len (int): 序列长度，默认值为20
        save_path (str, optional): 如果提供，则保存图像；否则显示图像，默认值为None
    """
    tisa = Tisa()  
    tisa(seq_len)  
    tisa.visualize(seq_len=seq_len, save_path=save_path)  

if __name__ == "__main__":
    
    # 使用固定种子生成并保存图像和参数 以便对照mindspore版本验证是否成功复刻。
    print("运行固定种子版本...")
    run_tisa_visualization_fixed()

    # 原代码主函数：使用随机种子生成并显示图像
    #print("运行随机种子版本...")
    #run_tisa_visualization_random()