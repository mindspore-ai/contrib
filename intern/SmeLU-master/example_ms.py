import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context, grad
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt


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

def main():

    fig, ax = plt.subplots(1, 1)
    fig_grad, ax_grad = plt.subplots(1, 1)


    for beta in [0.5, 1., 2., 3., 4.]:

        smelu = SmeLU(beta=beta)

        input_np = np.linspace(-6, 6, 1000).astype(np.float32)
        input_tensor = Tensor(input_np, mindspore.float32)

        def forward_fn(x):
            output = smelu(x)
            loss = output.sum()
            return loss


        output = smelu(input_tensor)


        grads = grad(forward_fn)(input_tensor)


        ax.plot(input_np, output.asnumpy(), label=str(beta))
        ax_grad.plot(input_np, grads.asnumpy(), label=str(beta))


    ax.legend()
    ax_grad.legend()
    ax.set_title("SmeLU_ms")
    ax_grad.set_title("SmeLU_ms gradient")
    ax.grid()
    ax_grad.grid()

    plt.show()

if __name__ == '__main__':
    main()
