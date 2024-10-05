import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

class DefConv(nn.Cell):
    def __init__(self, inc, outc, kernel_size = 3):
        super(DefConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_kernel = nn.Conv2d(
            inc, outc, kernel_size=kernel_size, stride=kernel_size)

    def construct(self, x, offset):
        
        assert offset.shape[1]//2 == self.kernel_size * self.kernel_size 
        x = ops.interpolate(x,(x.shape[2]*self.kernel_size,x.shape[3]*self.kernel_size))
        offset = self.reshape_flow(offset)
        x = self.flow_warp(x,offset)
        out = self.conv_kernel(x)
        
        return out


    def flow_warp(self, x, flow, padding_mode='border'):
        """Warp an image or feature map with optical flow
        Args:
            x (Tensor): size (n, c, h, w)
            flow (Tensor): size (n, 2, h, w), values range from -1 to 1
            padding_mode (str): 'zeros' or 'border'

        Returns:
            Tensor: warped image or feature map
        """
        assert x.shape[-2:] == flow.shape[-2:]
        n, _, h, w = x.shape
        x_ = ops.arange(w).view(1, -1).broadcast_to((h, -1))
        y_ = ops.arange(h).view(-1, 1).broadcast_to((-1, w))
        grid = ops.stack([x_, y_], axis=0).float()
        grid = grid.unsqueeze(0).broadcast_to((n, -1, -1, -1))
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid += 2 * flow
        grid = grid.permute(0, 2, 3, 1)
        return ops.grid_sample(x, grid, padding_mode=padding_mode)

    def reshape_flow(self, flow):
        # scale flow field for grid_sample to -1 1
        max_min = flow.max() - flow.min()
        flow = 2.*(flow - flow.min())/max_min-1
        # reshape offset for flow
        flow = flow.reshape(-1,2,flow.shape[2]*self.kernel_size, flow.shape[3]*self.kernel_size)
        return flow
        
def main():
    inc = 64
    outc = 64
    kernel_size = 3
    
    model = DefConv(inc, outc, kernel_size)
    print(model)
    params = list(model.trainable_params())
    total_params = sum(p.size for p in params if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    input_tensor = Tensor(np.random.randn(1, inc, 64, 64), ms.float32)
    offset_tensor = Tensor(np.random.randn(1, kernel_size * kernel_size * 2, 64, 64), ms.float32)
    output = model(input_tensor, offset_tensor)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()