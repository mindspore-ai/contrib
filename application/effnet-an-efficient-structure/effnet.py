import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

class Flatten(nn.Cell):
    def construct(self, x):
        x = x.view(x.shape[0], -1)
        return x
     
class EffNet(nn.Cell):
    def __init__(self, nb_classes=10, include_top=True, weights=None):
        super(EffNet, self).__init__()
        
        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.flatten = Flatten()
        self.dense = nn.Dense(4096, nb_classes)
        self.include_top = include_top
        self.weights = weights

    def make_layers(self, ch_in, ch_out):
        layers = [
            nn.Conv2d(3, ch_in, kernel_size=(1,1), stride=(1,1), has_bias=False, padding=0, dilation=(1,1)) if ch_in ==32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1,1),stride=(1,1), has_bias=False, padding=0, dilation=(1,1)) ,
            self.make_post(ch_in),
            # DepthWiseConvolution2D
            nn.Conv2d(ch_in, 1 * ch_in, group=ch_in, kernel_size=(1, 3),stride=(1,1), has_bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # DepthWiseConvolution2D
            nn.Conv2d(ch_in, 1 * ch_in, group=ch_in, kernel_size=(3, 1), stride=(1,1), has_bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), has_bias=False, dilation=(1,1)),
            self.make_post(ch_out),
        ]
        return nn.SequentialCell(*layers)


    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.dense(x)
        return x
    
if __name__ == '__main__':
    model = EffNet(nb_classes=10, include_top=True)
    input_shape = (1, 3, 32, 32)
    input_data = Tensor(np.random.rand(*input_shape), dtype=ms.float32)
    output = model(input_data)
    print("Output:", output)