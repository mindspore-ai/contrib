import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import time

class SmeLU(nn.Cell):


    def __init__(self, beta: float = 2.) -> None:

        super(SmeLU, self).__init__()
        assert beta >= 0., f"Beta必须大于等于零。给定的beta={beta}。"
        self.beta = beta

    def __repr__(self) -> str:

        return f"SmeLU(beta={self.beta})"

    def construct(self, input: Tensor) -> Tensor:

        zero = ops.zeros_like(input)
        beta_tensor = Tensor(self.beta, input.dtype)
        output = ops.where(input >= beta_tensor, input, zero)
        abs_input = ops.abs(input)
        condition = abs_input <= beta_tensor
        temp = ((input + beta_tensor) ** 2) / (4.0 * beta_tensor)
        output = ops.where(condition, temp, output)
        return output


def relu_runtime():
    network = nn.SequentialCell([nn.ReLU() for _ in range(100)])
    input = Tensor(np.random.randn(1, 3, 1024, 1024), mindspore.float32)

    def forward_fn(x):
        output = network(x)
        loss = output.sum()
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, grad_position=0)

    start_time = time.time()
    loss, grads = grad_fn(input)
    end_time = time.time()
    total_time = (end_time - start_time) * 1000  # 转换为毫秒
    print("ReLU (100x) 运行时间（前向和后向）:", total_time, "毫秒")


def smelu_runtime():
    network = nn.SequentialCell([SmeLU() for _ in range(100)])
    input = Tensor(np.random.randn(1, 3, 1024, 1024), mindspore.float32)

    def forward_fn(x):
        output = network(x)
        loss = output.sum()
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, grad_position=0)

    start_time = time.time()
    loss, grads = grad_fn(input)
    end_time = time.time()
    total_time = (end_time - start_time) * 1000  
    print("SmeLU (100x) 运行时间（前向和后向）:", total_time, "毫秒")


def main():
    relu_runtime()
    smelu_runtime()


if __name__ == '__main__':
    main()
