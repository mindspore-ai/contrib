import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeUniform


class ConvBlock(nn.Cell):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.block = nn.SequentialCell(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, pad_mode='pad',
                      weight_init=HeUniform()),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1, pad_mode='pad',
                      weight_init=HeUniform()),
            nn.LeakyReLU(alpha=0.2)
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0, pad_mode='pad',
                                weight_init=HeUniform())

    def construct(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class SSA(nn.Cell):
    def __init__(self, in_channel, strides=1):
        super(SSA, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=strides, padding=1, pad_mode='pad',
                               weight_init=HeUniform())
        self.relu1 = nn.LeakyReLU(alpha=0.2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=strides, padding=1, pad_mode='pad',
                               weight_init=HeUniform())
        self.relu2 = nn.LeakyReLU(alpha=0.2)
        self.conv11 = nn.Conv2d(in_channel, 16, kernel_size=1, stride=strides, padding=0, pad_mode='pad',
                                weight_init=HeUniform())

        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=3)
        self.reshape = ops.Reshape()
        self.matmul = ops.MatMul()
        self.matrix_inverse = ops.MatrixInverse()

    def construct(self, input1, input2):
        # Input1/2 shape: (N, C, H, W)
        input1 = self.transpose(input1, (0, 2, 3, 1))  # (N, H, W, C)
        input2 = self.transpose(input2, (0, 2, 3, 1))  # (N, H, W, C)
        cat = self.concat((input1, input2))  # (N, H, W, 2C)
        cat = self.transpose(cat, (0, 3, 1, 2))  # (N, 2C, H, W)

        out1 = self.relu1(self.conv1(cat))  # (N, 16, H, W)
        out1 = self.relu2(self.conv2(out1))  # (N, 16, H, W)
        out2 = self.conv11(cat)  # (N, 16, H, W)
        conv = out1 + out2  # (N, 16, H, W)

        conv = self.transpose(conv, (0, 2, 3, 1))  # (N, H, W, 16)
        N, H, W, K = conv.shape
        V = self.reshape(conv, (H * W, K))  # (H*W, 16)

        Vtrans = self.transpose(V, (1, 0))  # (16, H*W)
        VV = self.matmul(Vtrans, V)  # (16, 16)
        Vinverse = self.matrix_inverse(VV)  # (16, 16)
        Projection = self.matmul(self.matmul(V, Vinverse), Vtrans)  # (H*W, H*W)

        # Process input1
        H1, W1, C1 = input1.shape[1], input1.shape[2], input1.shape[3]
        X1 = self.reshape(input1, (H1 * W1, C1))  # (H*W, C1)
        Yproj = self.matmul(Projection, X1)  # (H*W, C1)
        Y = self.reshape(Yproj, (H1, W1, C1))  # (H, W, C1)
        Y = self.reshape(Y, (N, H1, W1, C1))  # (N, H, W, C1)
        Y = self.transpose(Y, (0, 3, 1, 2))  # (N, C1, H, W)
        return Y


class NBNet(nn.Cell):
    def __init__(self, num_classes=10):
        super(NBNet, self).__init__()
        # Encoder
        self.ConvBlock1 = ConvBlock(3, 32, strides=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip1 = nn.SequentialCell([ConvBlock(32, 32) for _ in range(4)])
        self.ssa1 = SSA(64)

        self.ConvBlock2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip2 = nn.SequentialCell([ConvBlock(64, 64) for _ in range(3)])
        self.ssa2 = SSA(128)

        self.ConvBlock3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip3 = nn.SequentialCell([ConvBlock(128, 128) for _ in range(2)])
        self.ssa3 = SSA(256)

        self.ConvBlock4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip4 = nn.SequentialCell(ConvBlock(256, 256))
        self.ssa4 = SSA(512)

        self.ConvBlock5 = ConvBlock(256, 512)

        # Decoder
        self.upv6 = nn.Conv2dTranspose(512, 256, kernel_size=2, stride=2)
        self.ConvBlock6 = ConvBlock(512, 256)

        self.upv7 = nn.Conv2dTranspose(256, 128, kernel_size=2, stride=2)
        self.ConvBlock7 = ConvBlock(256, 128)

        self.upv8 = nn.Conv2dTranspose(128, 64, kernel_size=2, stride=2)
        self.ConvBlock8 = ConvBlock(128, 64)

        self.upv9 = nn.Conv2dTranspose(64, 32, kernel_size=2, stride=2)
        self.ConvBlock9 = ConvBlock(64, 32)

        self.conv10 = nn.Conv2d(32, 3, kernel_size=3, padding=1, pad_mode='pad')
        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        # Encoder
        conv1 = self.ConvBlock1(x)  # (N,3,128,128) -> (N,32,128,128)
        pool1 = self.pool1(conv1)  # (N,32,64,64)

        conv2 = self.ConvBlock2(pool1)  # (N,64,64,64)
        pool2 = self.pool2(conv2)  # (N,64,32,32)

        conv3 = self.ConvBlock3(pool2)  # (N,128,32,32)
        pool3 = self.pool3(conv3)  # (N,128,16,16)

        conv4 = self.ConvBlock4(pool3)  # (N,256,16,16)
        pool4 = self.pool4(conv4)  # (N,256,8,8)

        conv5 = self.ConvBlock5(pool4)  # (N,512,8,8)

        # Decoder
        up6 = self.upv6(conv5)  # (N,256,16,16)
        skip4 = self.ssa4(self.skip4(conv4), up6)  # (N,256,16,16)
        up6 = self.cat((up6, skip4))  # (N,512,16,16)
        conv6 = self.ConvBlock6(up6)  # (N,256,16,16)

        up7 = self.upv7(conv6)  # (N,128,32,32)
        skip3 = self.ssa3(self.skip3(conv3), up7)  # (N,128,32,32)
        up7 = self.cat((up7, skip3))  # (N,256,32,32)
        conv7 = self.ConvBlock7(up7)  # (N,128,32,32)

        up8 = self.upv8(conv7)  # (N,64,64,64)
        skip2 = self.ssa2(self.skip2(conv2), up8)  # (N,64,64,64)
        up8 = self.cat((up8, skip2))  # (N,128,64,64)
        conv8 = self.ConvBlock8(up8)  # (N,64,64,64)

        up9 = self.upv9(conv8)  # (N,32,128,128)
        skip1 = self.ssa1(self.skip1(conv1), up9)  # (N,32,128,128)
        up9 = self.cat((up9, skip1))  # (N,64,128,128)
        conv9 = self.ConvBlock9(up9)  # (N,32,128,128)

        conv10 = self.conv10(conv9)  # (N,3,128,128)
        return x + conv10


if __name__ == "__main__":
    ms.set_context(device_target="CPU")

    model = NBNet()
    input = ms.Tensor(ms.numpy.randn(1, 3, 128, 128), dtype=ms.float32)
    output = model(input)
    # print(output)
    print(output.shape)  # (1, 3, 128, 128)

