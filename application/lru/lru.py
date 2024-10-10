import math
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, ops


class LRU(nn.Cell):
    def __init__(
        self, in_features, out_features, state_features, rmin=0, rmax=1, max_phase=6.283
    ):
        super().__init__()
        self.out_features = out_features
        self.D = Parameter(
            ops.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = ops.rand(state_features)
        u2 = ops.rand(state_features)
        self.nu_log = Parameter(
            ops.log(-0.5 * ops.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2))
        )
        self.theta_log = Parameter(ops.log(max_phase * u2))
        Lambda_mod = ops.exp(-ops.exp(self.nu_log))
        self.gamma_log = Parameter(
            ops.log(ops.sqrt(ops.ones_like(Lambda_mod) - ops.square(Lambda_mod)))
        )
        
        B_re = ops.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = ops.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B_re = Parameter(B_re)
        self.B_im = Parameter(B_im)
        
        C_re = ops.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = ops.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C_re = Parameter(C_re)
        self.C_im = Parameter(C_im)
        
        self.state_re = Parameter(ops.zeros(state_features, ms.float32))
        self.state_im = Parameter(ops.zeros(state_features, ms.float32))

    def construct(self, input, state=None):
        if state is not None:
            self.state_re, self.state_im = state
        else:
            self.state_re = ops.zeros_like(self.state_re)
            self.state_im = ops.zeros_like(self.state_im)

        Lambda_mod = ops.exp(-ops.exp(self.nu_log))
        Lambda_re = Lambda_mod * ops.cos(ops.exp(self.theta_log))
        Lambda_im = Lambda_mod * ops.sin(ops.exp(self.theta_log))
        gammas = ops.exp(self.gamma_log).unsqueeze(-1)
        output = ops.zeros(
            [i for i in input.shape[:-1]] + [self.out_features]
        )
        # Handle input of (Batches,Seq_length, Input size)
        if input.dim() == 3:
            for i, batch in enumerate(input):
                out_seq = ops.zeros((input.shape[1], self.out_features))
                for j, step in enumerate(batch):
                    new_state_re = Lambda_re * self.state_re - Lambda_im * self.state_im + gammas * self.B_re @ step.to(self.B_re.dtype)
                    new_state_im = Lambda_re * self.state_im + Lambda_im * self.state_re + gammas * self.B_im @ step.to(self.B_im.dtype)
                    self.state_re.set_data(new_state_re), self.state_im.set_data(new_state_im)
                    out_step = (self.C_re @ self.state_re - self.C_im @ self.state_im) + self.D @ step
                    out_seq[j] = out_step
                self.state_re = ops.zeros_like(self.state_re)
                self.state_im = ops.zeros_like(self.state_im)
                output[i] = out_seq
        # Handle input of (Seq_length, Input size)
        if input.dim() == 2:
            for i, step in enumerate(input):
                new_state_re = Lambda_re * self.state_re - Lambda_im * self.state_im + gammas * self.B_re @ step.to(self.B_re.dtype)
                new_state_im = Lambda_re * self.state_im + Lambda_im * self.state_re + gammas * self.B_im @ step.to(self.B_im.dtype)
                self.state_re.set_data(new_state_re), self.state_im.set_data(new_state_im)
                out_step = (self.C_re @ self.state_re - self.C_im @ self.state_im) + self.D @ step
                output[i] = out_step
            self.state_re = ops.zeros_like(self.state_re)
            self.state_im = ops.zeros_like(self.state_im)
        return output


if __name__ == '__main__':
    model = LRU(3, 2, 4)
    input = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
    output = model(input)
    print(output)
    print(output.shape)
