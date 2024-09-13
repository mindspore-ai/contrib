"""
The full model for the UColor
"""

from mindspore import nn, ops

from utils import conv2d, split


class SEBlock(nn.Cell):
    """
    SEBlock for Mindspore
    """
    def __init__(self, in_ch, out_ch, ratio):
        super(SEBlock, self).__init__()

        def gap(x):
            avg_pool = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]))
            return avg_pool(x)

        self.gap = gap
        mid_channel = int(out_ch / ratio)
        self.fc1 = nn.Dense(in_channels=in_ch, out_channels=mid_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(in_channels=mid_channel, out_channels=out_ch)
        self.sigmoid = ops.Sigmoid()
        self.out_dim = out_ch

    def construct(self, x):
        """
        SEBlock construct
        """
        squeeze = self.gap(x)
        squeeze = squeeze.reshape((x.shape[0], -1))
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        excitation = ops.reshape(excitation, (x.shape[0], self.out_dim, 1, 1))

        scale = x * excitation
        return scale


class BaseBlock(nn.Cell):
    """
    BaseBlock
    """
    def __init__(self, in_ch=3, out_ch=128):
        super(BaseBlock, self).__init__()
        self.relu = nn.ReLU()

        number_f = out_ch

        self.conv1 = conv2d(in_ch, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv2 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv3 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv4 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv5 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv6 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv7 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.conv8 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)

        self.max_pool_2x2 = nn.MaxPool2d(
            kernel_size=2, stride=2, pad_mode="valid")

    def construct(self, x):
        """
        BaseBlock construct
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.conv3(x3)

        add_x = x4 + x1

        x5 = self.relu(self.conv5(add_x))
        x6 = self.relu(self.conv5(x5))
        x7 = self.relu(self.conv5(x6))
        x8 = self.conv5(x7)

        return self.max_pool_2x2(x8 + x5), x1, x2, x3, x4, add_x, x5, x6, x7, x8, x8 + x5


class Ucolor(nn.Cell):
    """
    Main Model
    """
    def __init__(self):
        super(Ucolor, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool_2x2 = nn.MaxPool2d(
            kernel_size=2, stride=2, pad_mode="valid")
        self.cat = ops.Concat(axis=1)

        # encoder
        self.first_hsv_encoder = BaseBlock(3, 128)
        self.second_hsv_encoder = BaseBlock(128, 256)
        self.third_hsv_encoder = BaseBlock(256, 512)

        self.first_yuv_encoder = BaseBlock(3, 128)
        self.second_yuv_encoder = BaseBlock(128, 256)
        self.third_yuv_encoder = BaseBlock(256, 512)

        in_ch, out_ch, number_f = 3, 128, 128 * 3

        self.rgb_conv1_1 = conv2d(in_ch, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_2 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_3 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_4 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_5 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_6 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_7 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv1_8 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)

        in_ch, out_ch, number_f = 128, 256, 256 * 3

        self.rgb_conv2_1 = conv2d(in_ch, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_2 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_3 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_4 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_5 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_6 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_7 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv2_8 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)

        in_ch, out_ch, number_f = 256, 512, 512 * 3

        self.rgb_conv3_1 = conv2d(in_ch, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_2 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_3 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_4 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_5 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_6 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_7 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)
        self.rgb_conv3_8 = conv2d(number_f, out_ch, k_h=3, k_w=3, d_h=1, d_w=1)

        # concat
        self.seblock1 = SEBlock(in_ch=384, out_ch=384, ratio=16)
        self.seblock2 = SEBlock(in_ch=768, out_ch=768, ratio=16)
        self.seblock3 = SEBlock(in_ch=1536, out_ch=1536, ratio=16)

        self.cat_conv1 = conv2d(384, 128, k_h=3, k_w=3, d_h=1, d_w=1)
        self.cat_conv2 = conv2d(768, 256, k_h=3, k_w=3, d_h=1, d_w=1)
        self.cat_conv3 = conv2d(1536, 512, k_h=3, k_w=3, d_h=1, d_w=1)

        # decoder
        self.decoder3 = BaseBlock(512, 512)
        self.decoder2 = BaseBlock(768, 256)
        self.decoder1 = BaseBlock(384, 128)

        self.refine_conv = conv2d(128, 3, k_h=3, k_w=3, d_h=1, d_w=1)

    def construct(self, x):
        """
        Ucolor inference
        """
        rgb, hsv, lab, depth = split(x)
        depth2down = self.max_pool_2x2(1 - depth)
        depth4down = self.max_pool_2x2(depth2down)

        # first hsv encoder
        encoder1_down2_hsv, conv2_1_hsv, conv2_cb1_1_hsv, _, conv2_cb1_3_hsv, first_add_hsv, \
            conv2_2_hsv, conv2_cb1_4_hsv, conv2_cb1_5_hsv, _, first_add1_hsv \
            = self.first_hsv_encoder(hsv)

        # first yuv encoder
        encoder1_down2_yuv, conv2_1_yuv, conv2_cb1_1_yuv, _, conv2_cb1_3_yuv, first_add_yuv, \
            conv2_2_yuv, conv2_cb1_4_yuv, conv2_cb1_5_yuv, _, first_add1_yuv \
            = self.first_yuv_encoder(lab)

        # first RGB encoder
        conv2_1 = self.relu(self.rgb_conv1_1(rgb))
        conv2_cb1_1 = self.relu(self.rgb_conv1_2(self.cat([conv2_1, conv2_1_hsv, conv2_1_yuv])))
        conv2_cb1_2 = self.relu(
            self.rgb_conv1_3(self.cat([conv2_cb1_1, conv2_cb1_1_hsv, conv2_cb1_1_yuv])))
        conv2_cb1_3 = self.rgb_conv1_4(self.cat([conv2_cb1_2, conv2_cb1_3_hsv, conv2_cb1_3_yuv]))
        first_add = conv2_1 + conv2_cb1_3
        conv2_2 = self.relu(self.rgb_conv1_5(self.cat([first_add, first_add_hsv, first_add_yuv])))
        conv2_cb1_4 = self.relu(self.rgb_conv1_6(self.cat([conv2_2, conv2_2_hsv, conv2_2_yuv])))
        conv2_cb1_5 = self.relu(
            self.rgb_conv1_7(self.cat([conv2_cb1_4, conv2_cb1_4_hsv, conv2_cb1_4_yuv])))
        conv2_cb1_6 = self.rgb_conv1_8(self.cat([conv2_cb1_5, conv2_cb1_5_hsv, conv2_cb1_5_yuv]))
        first_add1 = conv2_2 + conv2_cb1_6
        encoder1_down2 = self.max_pool_2x2(first_add1)

        # second hsv encoder
        encoder2_down2_hsv, conv2_2_1_hsv, conv2_cb2_1_hsv, _, conv2_cb2_3_hsv, second_add_hsv, \
            conv2_2_2_hsv, conv2_cb2_4_hsv, conv2_cb2_5_hsv, _, second_add1_hsv \
            = self.second_hsv_encoder(encoder1_down2_hsv)

        # second yuv encoder
        encoder2_down2_yuv, conv2_2_1_yuv, conv2_cb2_1_yuv, _, conv2_cb2_3_yuv, second_add_yuv, \
            conv2_2_2_yuv, conv2_cb2_4_yuv, conv2_cb2_5_yuv, _, second_add1_yuv \
            = self.second_yuv_encoder(encoder1_down2_yuv)

        # second RGB encoder
        conv2_2_1 = self.relu(self.rgb_conv2_1(encoder1_down2))
        conv2_cb2_1 = self.relu(self.rgb_conv2_2(
            self.cat([conv2_2_1, conv2_2_1_hsv, conv2_2_1_yuv])))
        conv2_cb2_2 = self.relu(self.rgb_conv2_3(
            self.cat([conv2_cb2_1, conv2_cb2_1_hsv, conv2_cb2_1_yuv])))
        conv2_cb2_3 = self.rgb_conv2_4(
            self.cat([conv2_cb2_2, conv2_cb2_3_hsv, conv2_cb2_3_yuv]))
        second_add = conv2_2_1 + conv2_cb2_3
        conv2_2_2 = self.relu(self.rgb_conv2_5(
            self.cat([second_add, second_add_hsv, second_add_yuv])))
        conv2_cb2_4 = self.relu(self.rgb_conv2_6(
            self.cat([conv2_2_2, conv2_2_2_hsv, conv2_2_2_yuv])))
        conv2_cb2_5 = self.relu(self.rgb_conv2_7(
            self.cat([conv2_cb2_4, conv2_cb2_4_hsv, conv2_cb2_4_yuv])))
        conv2_cb2_6 = self.rgb_conv2_8(
            self.cat([conv2_cb2_5, conv2_cb2_5_hsv, conv2_cb2_5_yuv]))
        second_add1 = conv2_2_2 + conv2_cb2_6
        encoder2_down2 = self.max_pool_2x2(second_add1)

        # third hsv encoder
        _, conv2_3_1_hsv, conv2_cb3_1_hsv, conv2_cb3_2_hsv, _, third_add_hsv, \
            conv2_3_2_hsv, conv2_cb3_4_hsv, conv2_cb3_5_hsv, _, third_add1_hsv \
            = self.third_hsv_encoder(encoder2_down2_hsv)

        # third yuv encoder
        _, conv2_3_1_yuv, conv2_cb3_1_yuv, conv2_cb3_2_yuv, _, third_add_yuv, \
            conv2_3_2_yuv, conv2_cb3_4_yuv, conv2_cb3_5_yuv, _, third_add1_yuv \
            = self.third_yuv_encoder(encoder2_down2_yuv)

        # third RGB encoder
        conv2_3_1 = self.relu(self.rgb_conv3_1(encoder2_down2))
        conv2_cb3_1 = self.relu(self.rgb_conv3_2(
            self.cat([conv2_3_1, conv2_3_1_hsv, conv2_3_1_yuv])))
        conv2_cb3_2 = self.relu(self.rgb_conv3_3(
            self.cat([conv2_cb3_1, conv2_cb3_1_hsv, conv2_cb3_1_yuv])))
        conv2_cb3_3 = self.rgb_conv3_4(
            self.cat([conv2_cb3_2, conv2_cb3_2_hsv, conv2_cb3_2_yuv]))
        third_add = conv2_3_1 + conv2_cb3_3
        conv2_3_2 = self.relu(self.rgb_conv3_5(
            self.cat([third_add, third_add_hsv, third_add_yuv])))
        conv2_cb3_4 = self.relu(self.rgb_conv3_6(
            self.cat([conv2_3_2, conv2_3_2_hsv, conv2_3_2_yuv])))
        conv2_cb3_5 = self.relu(self.rgb_conv3_7(
            self.cat([conv2_cb3_4, conv2_cb3_4_hsv, conv2_cb3_4_yuv])))
        conv2_cb3_6 = self.rgb_conv3_8(
            self.cat([conv2_cb3_5, conv2_cb3_5_hsv, conv2_cb3_5_yuv]))
        third_add1 = conv2_3_2 + conv2_cb3_6

        # concat
        third_con = self.cat([third_add1, third_add1_hsv, third_add1_yuv])
        channle_weight_third_con_temp = self.seblock3(third_con)
        third_con_ff = self.relu(self.cat_conv3(channle_weight_third_con_temp))

        second_con = self.cat([second_add1, second_add1_hsv, second_add1_yuv])
        channle_weight_second_con_temp = self.seblock2(second_con)
        second_con_ff = self.relu(
            self.cat_conv2(channle_weight_second_con_temp))

        first_con = self.cat([first_add1, first_add1_hsv, first_add1_yuv])
        channle_weight_first_con_temp = self.seblock1(first_con)
        first_con_ff = self.relu(self.cat_conv1(channle_weight_first_con_temp))

        # first decoder
        decoder_input = third_con_ff + third_con_ff * depth4down
        first_dadd1 = self.decoder3(decoder_input)[-1]
        decoder1_down2 = ops.ResizeBilinear(
            [second_con_ff.shape[-2], second_con_ff.shape[-1]])(first_dadd1)

        # second decoder
        decoder_input1 = second_con_ff + second_con_ff * depth2down
        concat_1 = self.cat([decoder1_down2, decoder_input1])
        second_dadd1 = self.decoder2(concat_1)[-1]
        decoder2_down2 = ops.ResizeBilinear(
            [first_con_ff.shape[-2], first_con_ff.shape[-1]])(second_dadd1)

        # third decoder
        decoder_input2 = first_con_ff + first_con_ff * (1 - depth)
        concat_2 = self.cat([decoder2_down2, decoder_input2])
        third_dadd1 = self.decoder1(concat_2)[-1]

        conv2_refine = self.refine_conv(third_dadd1)

        return conv2_refine
