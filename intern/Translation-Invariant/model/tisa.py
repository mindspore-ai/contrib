import numpy as np
import mindspore
from mindspore import nn, ops
import matplotlib.pyplot as plt

class Tisa(nn.Cell):
    def __init__(self, num_attention_heads: int = 12, num_kernels: int = 5):
        super(Tisa, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kernels = num_kernels

        # Parameters to represent the kernels used in radial basis functions
        self.kernel_offsets = mindspore.Parameter(
            mindspore.Tensor(np.random.normal(0.0, 5.0, (self.num_kernels, self.num_attention_heads)), mindspore.float32),
            requires_grad=True
        )
        self.kernel_amplitudes = mindspore.Parameter(
            mindspore.Tensor(np.random.normal(0.1, 0.01, (self.num_kernels, self.num_attention_heads)), mindspore.float32),
            requires_grad=True
        )
        self.kernel_sharpness = mindspore.Parameter(
            mindspore.Tensor(np.random.normal(0.1, 0.01, (self.num_kernels, self.num_attention_heads)), mindspore.float32),
            requires_grad=True
        )

        # MindSpore operation for Gather
        self.gather = ops.Gather()

    def create_relative_offsets(self, seq_len: int):
        """Generates relative offsets for positional encodings from -seq_len+1 to seq_len-1."""
        # Ensure that relative_offsets is a Tensor, not ndarray
        return mindspore.Tensor(np.arange(-seq_len + 1, seq_len), mindspore.int32)

    def compute_positional_scores(self, relative_offsets):
        """Compute positional scores using radial basis functions for each attention head."""
        # Ensure that the kernel_offsets is a Tensor
        kernel_offsets_tensor = self.kernel_offsets.unsqueeze(-1)

        # Convert relative_offsets to mindspore.float32 if needed
        relative_offsets_tensor = mindspore.Tensor(relative_offsets, mindspore.float32)

        # Apply RBF to compute positional scores
        rbf_scores = (
            self.kernel_amplitudes.unsqueeze(-1) *
            ops.Exp()(-ops.Abs()(kernel_offsets_tensor - relative_offsets_tensor) ** 2)
        ).sum(axis=0)
        return rbf_scores

    def scores_to_toeplitz_matrix(self, positional_scores, seq_len: int):
        """Converts positional scores to a Toeplitz matrix format."""
        # Generate a Toeplitz matrix
        deformed_toeplitz = (
            (mindspore.Tensor(np.arange(0, -(seq_len ** 2), step=-1) + (seq_len - 1), mindspore.int32).view(seq_len, seq_len))
            + (seq_len + 1) * mindspore.Tensor(np.arange(seq_len), mindspore.int32).view(-1, 1)
        ).view(-1)

        # Map positional scores to the flattened Toeplitz matrix using gather
        expanded_positional_scores = self.gather(positional_scores, deformed_toeplitz, 1)  # axis=1 -> dim=1
        return expanded_positional_scores.view(self.num_attention_heads, seq_len, seq_len)

    def construct(self, seq_len: int):
        """Computes the positional contribution to the attention matrix."""
        if self.num_kernels == 0:
            return mindspore.Tensor(np.zeros((self.num_attention_heads, seq_len, seq_len)), mindspore.float32)

        relative_offsets = self.create_relative_offsets(seq_len)
        positional_scores_vector = self.compute_positional_scores(relative_offsets)
        positional_scores_matrix = self.scores_to_toeplitz_matrix(positional_scores_vector, seq_len)
        return positional_scores_matrix

    def visualize(self, seq_len: int = 10, attention_heads=None):
        """Visualizes the positional scores for the given attention heads."""
        if attention_heads is None:
            attention_heads = list(range(self.num_attention_heads))

        x = self.create_relative_offsets(seq_len).asnumpy()
        y = self.compute_positional_scores(x).asnumpy()

        # Plot the positional scores for the selected attention heads
        for i in attention_heads:
            plt.plot(x, y[i], label=f'Head {i}')
        plt.xlabel('Relative Offset')
        plt.ylabel('Positional Score')
        plt.legend()
        plt.show()


def main():
    tisa = Tisa()
    tisa(20)
    tisa.visualize(seq_len=20)


if __name__ == "__main__":
    main()
    