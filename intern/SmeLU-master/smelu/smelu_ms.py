from mindspore import nn, Tensor, ops
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