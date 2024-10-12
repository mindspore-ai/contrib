import mindspore
from mindspore import Tensor, nn, ops
from functools import partial
import numpy as np

class OctConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, alpha_in=0.5, alpha_out=0.5):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out, self.kernel_size = alpha_in, alpha_out, kernel_size
        self.H2H, self.L2L, self.H2L, self.L2H = None, None, None, None
        
        if not (alpha_in == 0.0 and alpha_out == 0.0):
            self.L2L = nn.Conv2d(int(alpha_in * in_channels),
                                 int(alpha_out * out_channels),
                                 kernel_size, stride, pad_mode='pad', padding=kernel_size//2)
        if not (alpha_in == 0.0 and alpha_out == 1.0):
            self.L2H = nn.Conv2d(int(alpha_in * in_channels),
                                 out_channels - int(alpha_out * out_channels),
                                 kernel_size, stride, pad_mode='pad', padding=kernel_size//2)
        if not (alpha_in == 1.0 and alpha_out == 0.0):
            self.H2L = nn.Conv2d(in_channels - int(alpha_in * in_channels),
                                 int(alpha_out * out_channels),
                                 kernel_size, stride, pad_mode='pad', padding=kernel_size//2)
        if not (alpha_in == 1.0 and alpha_out == 1.0):
            self.H2H = nn.Conv2d(in_channels - int(alpha_in * in_channels),
                                 out_channels - int(alpha_out * out_channels),
                                 kernel_size, stride, pad_mode='pad', padding=kernel_size//2)
        
        self.upsample = Upsample(None, 3, 'nearest')
        self.avg_pool = partial(mindspore.ops.avg_pool2d, kernel_size=kernel_size, stride=kernel_size)

    def construct(self, x):
        hf, lf = x
        h2h, l2l, h2l, l2h = None, None, None, None
        
        if self.H2H is not None:
            h2h = self.H2H(hf)
        if self.L2L is not None:
            l2l = self.L2L(lf)
        if self.H2L is not None:
            h2l = self.H2L(self.avg_pool(hf))
        if self.L2H is not None:
            l2h = self.upsample(self.L2H(lf))
        
        hf_, lf_ = 0, 0
        for i in [h2h, l2h]:
            if i is not None:
                hf_ = hf_ + i
                
        for i in [l2l, h2l]:
            if i is not None:
                lf_ = lf_ + i

        return hf_, lf_

class Upsample(nn.Cell):

    def __init__(self, sizes=None, scales=None, mode="nearest"):
        super(Upsample, self).__init__()
        self.sizes = sizes
        self.scales = scales
        self.mode = mode

    def construct(self, x):
        return ops.ResizeNearestNeighbor((x.shape[-2] * self.scales, x.shape[-1] * self.scales))(x)

def main():

    in_channels = 16      
    out_channels = 32     
    kernel_size = 3       
    stride = 1            
    alpha_in = 0.5        
    alpha_out = 0.5       

    oct_conv = OctConv(in_channels, out_channels, kernel_size, stride, alpha_in, alpha_out)

    hf_input = mindspore.ops.randn(1, 8, 96, 96)
    lf_input = mindspore.ops.randn(1, 8, 32, 32) 
    x = (hf_input, lf_input)

    hf_output, lf_output = oct_conv(x)

    # 输出结果
    print("High Frequency Output Shape:", hf_output.shape)
    print("Low Frequency Output Shape:", lf_output.shape)

if __name__ == "__main__":
    main()
