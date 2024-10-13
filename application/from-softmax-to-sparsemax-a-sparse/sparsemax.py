# Sparsemax implementation in MindSpore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

class Sparsemax(nn.Cell):
    """Sparsemax activation function."""

    def __init__(self, dim=None):
        """Initialize Sparsemax activation.

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()
        self.dim = -1 if dim is None else dim

    def construct(self, input):
        """Construct function.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying Sparsemax.
        """
        # Reshape input for processing
        input = ops.swapaxes(input, 0, self.dim)
        original_size = input.shape
        input = input.reshape((input.shape[0], -1))
        input = ops.transpose(input, (1, 0))
        dim = 1

        number_of_logits = input.shape[dim]

        # Numerical stability
        max_vals = ops.ReduceMax(keep_dims=True)(input, axis=dim)
        input = input - max_vals

        # Sort input in descending order
        zs, _ = ops.Sort(axis=dim, descending=True)(input)

        # Create range tensor
        range = mnp.arange(1, number_of_logits + 1, dtype=input.dtype)
        range = range.reshape((1, -1))
        range = ops.broadcast_to(range, zs.shape)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = ops.cumsum(zs, axis=dim)
        is_gt = ops.gt(bound, cumulative_sum_zs)
        is_gt = ops.cast(is_gt, input.dtype)
        k = ops.ReduceMax(keep_dims=True)(is_gt * range, axis=dim)

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (ops.ReduceSum(keep_dims=True)(zs_sparse, axis=dim) - 1) / k
        taus = ops.broadcast_to(taus, input.shape)

        # Apply Sparsemax
        output = ops.maximum(ops.zeros_like(input), input - taus)

        # Reshape back to original
        output = ops.transpose(output, (1, 0))
        output = output.reshape(original_size)
        output = ops.swapaxes(output, 0, self.dim)

        return output