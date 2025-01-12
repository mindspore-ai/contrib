import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops

class Generator(nn.Cell):
    def __init__(self, input_nc=3, output_nc=4, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7,padding=0,pad_mode='pad'),
                 norm_layer(ngf),
                 nn.ReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1,pad_mode='pad'),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU()]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                kernel_size=3, stride=1,padding=0, pad_mode='pad'),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(),
                      nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2)*4,
                                kernel_size=1, stride=1,padding=0, pad_mode='pad'),
                      nn.PixelShuffle(2),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(),
                     ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7,padding=0, pad_mode='pad')]

        self.model = nn.SequentialCell(*model)

    def construct(self, input):
        output = self.model(input)
        mask = ops.sigmoid(output[:, :1])
        oimg = output[:, 1:]

        multiples = (1, 3, 1, 1)
        mask_repeated = ops.Tile()(mask, multiples)
        mask = mask_repeated
        oimg = oimg*mask + input*(1-mask)
        return oimg, mask

class ResnetBlock(nn.Cell):
    def __init__(self, dim, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout)

    def build_conv_block(self, dim, norm_layer, use_dropout):
        conv_block = [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3,padding=0, pad_mode='pad'),
                       norm_layer(dim),
                       nn.ReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(p=0.5)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3,padding=0, pad_mode='pad'),
                       norm_layer(dim)]

        return nn.SequentialCell(*conv_block)

    def construct(self, x):
        conv_out = self.conv_block(x)
        out = x + conv_out
        return out


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_tower = nn.SequentialCell(
            nn.Conv2d(3,   64,  4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,  128,  4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,  256,  4, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,  512, 4, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 4),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1, 1),
        )

    def construct(self, img):
        output = self.conv_tower(img)
        return output
