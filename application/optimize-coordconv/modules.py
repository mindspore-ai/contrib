from mindspore import Parameter, ops
from mindspore.nn import Cell

class CoordXY(Cell):
    """
    This operation next to convolution provides CoordConv layer result from paper https://arxiv.org/pdf/1807.03247.pdf.
    
    Parameters:
    -----------
    out_channels : int
        number of output channels
    std : float
        standard deviation for initialization of parameters
    """
    def __init__(self, out_channels : int, std : float=0.1):
        super(CoordXY, self).__init__()
        self.out_channels = out_channels
        self.std = std
        self.alpha = Parameter((ops.randn(out_channels) * self.std).view(1, self.out_channels, 1, 1))
        self.beta = Parameter((ops.randn(out_channels) * self.std).view(1, self.out_channels, 1, 1))
    
    def construct(self, input):
        assert len(input.shape) == 4, "Tensor should have 4 dimensions"
        dimx = input.shape[-1]
        dimy = input.shape[-2]
        x_special = ops.linspace(-1, 1, steps=dimx).view(1, 1, 1, dimx).broadcast_to(input.shape)
        y_special = ops.linspace(-1, 1, steps=dimy).view(1, 1, dimy, 1).broadcast_to(input.shape)
        ops.assign_add(input, self.alpha * x_special)
        ops.assign_add(input, self.beta * y_special)
        return input

if __name__ == "__main__":
    coordxy = CoordXY(out_channels=3, std=0.1)
    input_tensor = ops.randn(1, 3, 5, 5)
    output = coordxy(input_tensor)
    print(output.shape)
