import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

class conv_block(nn.Cell):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,has_bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,has_bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )


    def construct(self,x):
        x = self.conv(x)
        return x
        
class SqueezeAttentionBlock(nn.Cell):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2.0, recompute_scale_factor=True)

    def construct(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        # print(y.shape)
        y = self.upsample(y)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y
    
def generate_random_input(batch_size, channels, height, width):
    return ms.Tensor(np.random.rand(batch_size, channels, height, width).astype(np.float32))

def main():
    ms.set_seed(0)
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    input_data = generate_random_input(batch_size, channels, height, width)
    attention_block = SqueezeAttentionBlock(channels, 64)
    output = attention_block(input_data)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()