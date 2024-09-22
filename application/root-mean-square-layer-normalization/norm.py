import numbers
import mindspore
from mindspore import nn, ops, Parameter, Tensor
import numpy as np
from typing import Union, List, Tuple

class RMSNorm(nn.Cell):
    def __init__(self, normalized_shape: Union[int, List[int], Tuple[int]], eps: float = 1e-5, bias: bool = False) -> None:
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(self.normalized_shape), mindspore.float32), name='weight')
        
        if bias:
            self.bias = Parameter(Tensor(np.zeros(self.normalized_shape), mindspore.float32), name='bias')
        else:
            self.bias = None

        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.rsqrt = ops.Rsqrt()

    def construct(self, input: Tensor) -> Tensor:
        var = self.reduce_mean(input ** 2, -1) + self.eps
        input_norm = input * self.rsqrt(var)
        weight_broadcast = ops.BroadcastTo(input.shape)(self.weight)

        rmsnorm = weight_broadcast * input_norm
        
        if self.bias is not None:
            bias_broadcast = ops.BroadcastTo(input.shape)(self.bias)
            rmsnorm = rmsnorm + bias_broadcast

        return rmsnorm

if __name__ == '__main__':
    input_data = Tensor(np.random.rand(2, 3, 4).astype(np.float32))
    rmsnorm_layer = RMSNorm(normalized_shape=4, bias=True)
    output = rmsnorm_layer(input_data)
    
    print("Input:\n", input_data.asnumpy())
    print("Output:\n", output.asnumpy())
