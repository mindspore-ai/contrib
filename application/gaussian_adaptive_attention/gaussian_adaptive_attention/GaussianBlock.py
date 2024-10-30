import mindspore
from mindspore import nn
from mindspore import Parameter, Tensor
import mindspore.ops as ops
from mindspore import dtype as mstype

class GaussianAdaptiveAttention(nn.Cell):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value, mean_offset_init=0, eps=1e-8):
        super(GaussianAdaptiveAttention, self).__init__()
        if not isinstance(norm_axis, int):
            raise ValueError("norm_axis must be an integer.")
        if num_heads <= 0 or not isinstance(num_heads, int):
            raise ValueError("num_heads must be a positive integer.")
        if num_gaussians <= 0 or not isinstance(num_gaussians, int):
            raise ValueError("num_gaussians must be a positive integer.")

        self.norm_axis = norm_axis
        self.eps = eps
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.num_gaussians = num_gaussians

        self.mean_offsets = Parameter(ops.zeros(num_gaussians, dtype=mstype.float32))
        self.c = Parameter(ops.exp(ops.randn(num_gaussians, dtype=mstype.float32)))

    def construct(self, x, return_attention_details=False):
        if x.dim() < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {x.dim()}.")
        if self.norm_axis >= x.dim() or self.norm_axis < -x.dim():
            raise ValueError(f"norm_axis {self.norm_axis} is out of bounds for input tensor with {x.dim()} dimensions.")

        mask = x != self.padding_value if self.padding_value is not None else None
        x_masked = ops.where(mask, x, ops.zeros_like(x)) if mask is not None else x

        mean = x_masked.mean(axis=self.norm_axis, keep_dims=True)
        var = x_masked.var(axis=self.norm_axis, keepdims=True) + self.eps

        mixture = 1
        for i in range(self.num_gaussians):
            adjusted_mean = mean + self.mean_offsets[i]
            y_norm = (x - adjusted_mean) / ops.sqrt(var)
            gaussian = ops.exp(-((y_norm ** 2) / (2.0 * (self.c[i] ** 2)))) / ops.sqrt(2 * mindspore.numpy.pi * (self.c[i] ** 2))
            mixture *= gaussian

        mixture /= mixture.sum(axis=self.norm_axis, keepdims=True).clamp(min=self.eps)

        if return_attention_details:
            return ops.where(mask, x * mixture, x) if mask is not None else x * mixture, mixture.detach()
        else:
            return ops.where(mask, x * mixture, x) if mask is not None else x * mixture
            
            
class MultiHeadGaussianAdaptiveAttention(nn.Cell):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value=None, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.attention_heads = nn.CellList([
            GaussianAdaptiveAttention(norm_axis, num_heads, num_gaussians, padding_value, eps)
            for _ in range(num_heads)
        ])

    def construct(self, x, return_attention_details=False):
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        if chunk_size == 0:
            raise ValueError(f"Input tensor size along norm_axis ({self.norm_axis}) must be larger than the number of heads ({self.num_heads}).")

        outputs, attention_details_ = [], []
        for i in range(self.num_heads):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            chunk = x.narrow(self.norm_axis, start_index, end_index - start_index)
            if return_attention_details:
                out, mixture = self.attention_heads[i](chunk, return_attention_details=True)
                outputs.append(out)
                attention_details_.append(mixture)
            else:
                outputs.append(self.attention_heads[i](chunk))

        if return_attention_details:
            return ops.cat(outputs, axis=self.norm_axis), ops.cat(attention_details_, axis=self.norm_axis)
        else:
            return ops.cat(outputs, axis=self.norm_axis)
            
            

class GaussianBlock(nn.Cell):
    def __init__(self, norm_axes, num_heads, num_gaussians, num_layers, padding_value=None, eps=1e-8):
        super().__init__()
        if len(norm_axes) != num_layers or len(num_heads) != num_layers or len(num_gaussians) != num_layers:
            raise ValueError("Lengths of norm_axes, num_heads, and num_gaussians must match num_layers.")

        self.layers = nn.CellList([
            MultiHeadGaussianAdaptiveAttention(norm_axes[i], num_heads[i], num_gaussians[i], padding_value, eps)
            for i in range(num_layers)
        ])

    def construct(self, x, return_attention_details=False):
        attention_details_ = {}
        for idx, layer in enumerate(self.layers):
            if return_attention_details:
                x_, attention_details = layer(x, return_attention_details=True)
                attention_details_['layer_'+str(idx)] = attention_details
                x = x_ + x
            else:
                x = layer(x) + x

        if return_attention_details:
            return x, attention_details_
        return x
