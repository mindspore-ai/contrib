import mindspore
from mindspore import nn, ops, Tensor

class DefConv(nn.Cell):
    def __init__(self, inc, outc, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=1)

    def construct(self, x, offset):
        assert offset.shape[1] // 2 == self.kernel_size * self.kernel_size
        new_size = (x.shape[2] * self.kernel_size, x.shape[3] * self.kernel_size)
        x = ops.interpolate(x, size=new_size)
        offset = self.reshape_flow(offset)
        x = self.flow_warp(x, offset)
        out = self.conv_kernel(x)
        return out

    def flow_warp(self, x, flow, padding_mode='border'):
        assert x.shape[-2:] == flow.shape[-2:]
        n, _, h, w = x.shape
        x_ = ops.arange(0, w).reshape(1, -1).broadcast_to((h, -1))
        y_ = ops.arange(0, h).reshape(-1, 1).broadcast_to((-1, w))
        grid = ops.stack([x_, y_], axis=0).float()
        grid = grid.expand_dims(0).broadcast_to((n, -1, -1, -1))
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid += 2 * flow
        grid = grid.permute((0, 2, 3, 1))
        return ops.grid_sample(x, grid, padding_mode=padding_mode)

    def reshape_flow(self, flow):
        max_min = flow.max() - flow.min()
        flow = 2. * (flow - flow.min()) / max_min - 1
        flow = flow.reshape(-1, 2, flow.shape[2] * self.kernel_size, flow.shape[3] * self.kernel_size)
        return flow

if __name__ == '__main__':
    model = DefConv(inc=3, outc=16, kernel_size=3)
    x = Tensor(ops.randn(1, 3, 32, 32), mindspore.float32)
    offset = Tensor(ops.randn(1, 2*3*3, 32, 32), mindspore.float32)
    output = model(x, offset)
    print("output=", output)
    print("output.shape=",output.shape)