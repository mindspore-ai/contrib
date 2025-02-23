import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, ops, Parameter, context
from mindspore.common.initializer import Normal, initializer


class Tisa(nn.Cell):
    def __init__(self, num_attention_heads: int = 12, num_kernels: int = 5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kernels = num_kernels

        # 参数初始化
        self.kernel_offsets = Parameter(
            initializer(Normal(sigma=5.0), (self.num_kernels, self.num_attention_heads)),
            name="kernel_offsets"
        )
        self.kernel_amplitudes = Parameter(
            initializer(Normal(mean=0.1, sigma=0.01), (self.num_kernels, self.num_attention_heads)),
            name="kernel_amplitudes"
        )
        self.kernel_sharpness = Parameter(
            initializer(Normal(mean=0.1, sigma=0.01), (self.num_kernels, self.num_attention_heads)),
            name="kernel_sharpness"
        )

    def create_relative_offsets(self, seq_len: int):
        """创建相对位置偏移量"""
        return ops.arange(-seq_len, seq_len + 1, dtype=ms.float32)

    def compute_positional_scores(self, relative_offsets):
        """计算位置得分"""
        # 维度扩展
        offsets_exp = self.kernel_offsets.expand_dims(-1)
        sharpness_exp = self.kernel_sharpness.expand_dims(-1)

        # RBF计算
        rbf_scores = (
            self.kernel_amplitudes.expand_dims(-1) *
            ops.exp(
                -ops.abs(sharpness_exp) *
                ((offsets_exp - relative_offsets) ** 2)
            )
        )
        return rbf_scores.sum(axis=0)

    def scores_to_toeplitz_matrix(self, positional_scores, seq_len: int):
        """转换为Toeplitz矩阵"""
        # 生成基础索引矩阵
        base_indices = ops.arange(seq_len).reshape(-1, 1)

        # 构建变形Toeplitz索引
        deformed_toeplitz = (
            (ops.arange(0, -(seq_len ** 2), -1, dtype=ms.int32) + (seq_len - 1))
            .reshape(seq_len, seq_len) +
            (seq_len + 1) * base_indices
        )

        # 收集分数
        expanded_scores = ops.gather(
            positional_scores,
            deformed_toeplitz.reshape(-1).astype(ms.int32),
            1
        )
        return expanded_scores.reshape(self.num_attention_heads, seq_len, seq_len)

    def construct(self, seq_len: int):
        """前向传播"""
        if self.num_kernels == 0:
            return ops.zeros((self.num_attention_heads, seq_len, seq_len), ms.float32)

        offsets = self.create_relative_offsets(seq_len)
        pos_scores = self.compute_positional_scores(offsets)
        return self.scores_to_toeplitz_matrix(pos_scores, seq_len)

    def visualize(self, seq_len: int = 10, attention_heads=None):
        """可视化结果"""
        if attention_heads is None:
            attention_heads = list(range(self.num_attention_heads))

        # 转换为numpy数组
        x = self.create_relative_offsets(seq_len).asnumpy()
        y = self.compute_positional_scores(
            self.create_relative_offsets(seq_len)
        ).asnumpy()

        # 绘制结果
        plt.figure(figsize=(12, 6))
        for i in attention_heads:
            plt.plot(x, y[i], label=f'Head {i}')
        plt.xlabel('Relative Position')
        plt.ylabel('Positional Score')
        plt.title('TISA Positional Scores')
        plt.legend()
        plt.show()


def main():
    # 设置运行模式
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Tisa()
    output = net(20)
    print("Output shape:", output.shape)
    net.visualize(seq_len=20)


if __name__ == "__main__":
    main()