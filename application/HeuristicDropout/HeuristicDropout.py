import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
import numpy as np
from mindspore.nn.probability.distribution import Categorical

# HeuristicDropout
class HeuristicDropout(nn.Cell):
    def __init__(self, rate=0.1, threshold=0.5, bin_count=10,training = True):
        super().__init__()
        self.rate = rate
        self.threshold = threshold
        self.bin_count = bin_count
        self.training = training
    def construct(self, x):
        if self.training:
            b, c, h, w = x.shape
            x_tanh = ops.Tanh()(x)
            xtype = x.dtype
            filter_identity = mindspore.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=xtype)
            filter_laplace = mindspore.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=xtype)
            var = ops.var(x, axis=(2, 3))
            quantize = ops.Round()(x_tanh * self.bin_count).view(b, c, -1)
            temp = mindspore.tensor(quantize.unsqueeze(-1) == ops.arange(0, self.bin_count+1),dtype=mindspore.float32)
            hist = ops.count_nonzero(temp, axis=2)
            hist = hist.astype(mindspore.float32)
            hist = hist + 1
            probabilities = (hist/(h*w+self.bin_count+1+1e-7)).astype(mindspore.float32)
            entropy = Categorical(dtype=mindspore.float32).entropy(probabilities)
            if len(entropy.shape) == 1:
                entropy = ops.expand_dims(entropy, 1)
            _, indices = ops.sort(entropy + 2. / (var + 1e-7), axis=1, descending=True)

            filters = ops.tile(filter_identity, (b * c, 1, 1, 1))
            temp2 = ops.tile((ops.arange(0, b) * c), (c, 1)).transpose(1, 0)
            indices_ = indices + temp2
            indices_ = ops.flatten(indices_[:, :round(self.rate * c)])
            filters[indices_, 0] = filter_laplace
            
            split_result = ops.split(x, 1, axis=0)
            x_ = ops.concat(split_result, axis=1)
            x_ = x_.astype(mindspore.float32)
            filters = filters.astype(mindspore.float32)
            outx_ = ops.Conv2D(out_channel=filters.shape[0], kernel_size=filters.shape[-1],
                       pad_mode='pad', pad=1, group=b * c)(x_,filters)
            outx = outx_.reshape(c, b, h, w).transpose(1, 0, 2, 3)
            return outx
        else:
            return x

# 自定义舍入

def custom_round_logic(input_):
    return ops.Round()(input_)

class AlternativeRound(ops.Custom):
    def __init__(self):
        super().__init__(custom_round_logic)

    def forward(self, input_):
        ctx = self.get_context()
        ctx.input = input_
        return custom_round_logic(input_)

    def backward(self, grd_output):
        grad_input = grd_output.copy()
        return grad_input


class HeuristicDropoutWithAlternativeRound(nn.Cell):
    def __init__(self, rate=0.1, threshold=0.5):
        super().__init__()
        self.rate = rate
        self.threshold = threshold
        self.bin_count = 10

    def construct(self, x):
        if self.training:
            b, c, h, w = x.shape
            x_tanh = ops.Tanh()(x)
            xtype = x.dtype
            filter_identity = mindspore.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=xtype)
            filter_laplace = mindspore.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=xtype)
            var = ops.var(x, axis=(2, 3))
            alternative_round = AlternativeRound()
            quantize = alternative_round(x_tanh * self.bin_count).view(b, c, -1)
            temp = mindspore.tensor(quantize.unsqueeze(-1) == ops.arange(0, self.bin_count+1),dtype=mindspore.float32)
            hist = ops.count_nonzero(temp, axis=2)
            hist = hist.astype(mindspore.float32)
            hist = hist + 1
            probabilities = (hist/(h*w+self.bin_count+1+1e-7)).astype(mindspore.float32)
            print(probabilities.shape)
            entropy = Categorical(dtype=mindspore.float32).entropy(probabilities)
            if len(entropy.shape) == 1:
                entropy = ops.expand_dims(entropy, 1)
            _, indices = ops.sort(entropy + 2. / (var + 1e-7), axis=1, descending=True)

            filters = ops.tile(filter_identity, (b * c, 1, 1, 1))
            temp2 = ops.tile((ops.arange(0, b) * c), (c, 1)).transpose(1, 0)
            indices_ = indices + temp2
            indices_ = ops.flatten(indices_[:, :round(self.rate * c)])
            filters[indices_, 0] = filter_laplace
            
            split_result = ops.split(x, 1, axis=0)
            x_ = ops.concat(split_result, axis=1)
            x_ = x_.astype(mindspore.float32)
            filters = filters.astype(mindspore.float32)
            outx_ = ops.Conv2D(out_channel=filters.shape[0], kernel_size=filters.shape[-1],
                       pad_mode='pad', pad=1, group=b * c)(x_,filters)
            outx = outx_.reshape(c, b, h, w).transpose(1, 0, 2, 3)
            return outx
        else:
            return x
        
