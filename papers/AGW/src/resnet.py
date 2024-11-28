# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ResNet."""

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same')


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, pad_mode='same')


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=stride, padding=0, pad_mode='same')


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.997,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.
    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)
        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)
        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell(
                [_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])
        self.add = P.Add()

    def construct(self, x):
        """
        function of constructing
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.
    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError(
                "the length of layer_num, in_channels, out_channels list must be 4!")
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.
        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.
        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        function of constructing
        """
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c5


class ResNetSpecific(nn.Cell):
    """
    ResNet architecture.
    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 layer_nums,
                 in_channels,
                 out_channels,
                 ):
        super(ResNetSpecific, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError(
                "the length of layer_num, in_channels, out_channels list must be 4!")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5,
                                  gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNetShare(nn.Cell):
    """
    ResNet architecture.
    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides):
        super(ResNetShare, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError(
                "the length of layer_num, in_channels, out_channels list must be 4!")
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.
        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.
        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet50(pretrain="", last_conv_stride=1):
    """
    Get ResNet50 neural network.
    Returns:
        Cell, cell instance of ResNet50 neural network.
    Examples:
        >>> net = resnet50()
    """
    stride = [1, 2, 2, last_conv_stride]
    resnet = ResNet(ResidualBlock,
                    [3, 4, 6, 3],
                    [64, 256, 512, 1024],
                    [256, 512, 1024, 2048],
                    stride)

    if pretrain:
        param_dict = load_checkpoint(pretrain)
        load_param_into_net(resnet, param_dict)
        print('resnet-50 pretrained weight loaded!')

    return resnet


def resnet50_share(pretrain=""):
    """
    Get ResNet50 neural network.
    Returns:
        Cell, cell instance of ResNet50 neural network.
    Examples:
        >>> net = resnet50()
    """
    resnet = ResNetShare(ResidualBlock,
                         [3, 4, 6, 3],
                         [64, 256, 512, 1024],
                         [256, 512, 1024, 2048],
                         [1, 2, 2, 2])

    if pretrain:
        param_dict = load_checkpoint(pretrain)
        load_param_into_net(resnet, param_dict)

    return resnet


def resnet50_specific(pretrain=""):
    """
    Get ResNet50 neural network.
    Returns:
        Cell, cell instance of ResNet50 neural network.
    Examples:
        >>> net = resnet50()
    """
    resnet = ResNetSpecific([3, 4, 6, 3],
                            [64, 256, 512, 1024],
                            [256, 512, 1024, 2048],
                            )
    if pretrain:
        param_dict = load_checkpoint(pretrain)
        load_param_into_net(resnet, param_dict)

    return resnet


class NonLocal(nn.Cell):
    """
    class of non local
    """

    def __init__(self, in_channels, reduc_ratio=2):
        super(NonLocal, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio
        self.g = nn.SequentialCell([
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        ])
        self.weight = nn.SequentialCell([
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels, gamma_init='zeros',
                           beta_init='zeros')
        ])
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.bmm = P.BatchMatMul()
        self.trans = P.Transpose()
        self.print = P.Print()

    def construct(self, x):
        """
        x: (bs, c, h, w)
        """
        batch_size = x.shape[0]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = self.trans(g_x, (0, 2, 1))

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = self.trans(theta_x, (0, 2, 1))
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = self.bmm(theta_x, phi_x)
        n_shape = f.shape[-1]
        f_div_c = f / n_shape

        y = self.bmm(f_div_c, g_x)
        y = self.trans(y, (0, 2, 1))

        y = y.view(batch_size, self.inter_channels, *x.shape[2:])
        wy = self.weight(y)
        z = wy + x

        return z


class ResNetNL(nn.Cell):
    """
    Backbone net
    """

    def __init__(self, pretrain="", non_layers=None, last_conv_stride=1):
        super(ResNetNL, self).__init__()

        if not non_layers:
            non_layers = [0, 2, 3, 0]

        self.base_resnet = resnet50(pretrain, last_conv_stride)
        self.layers = [3, 4, 6, 3]

        self.nonlocal_1 = nn.CellList(
            [NonLocal(256) for i in range(non_layers[0])])
        self.nonlocal_1_idx = sorted(
            [self.layers[0] - (i + 1) for i in range(non_layers[0])])
        self.nonlocal_2 = nn.CellList(
            [NonLocal(512) for i in range(non_layers[1])])
        self.nonlocal_2_idx = sorted(
            [self.layers[1] - (i + 1) for i in range(non_layers[1])])
        self.nonlocal_3 = nn.CellList(
            [NonLocal(1024) for i in range(non_layers[2])])
        self.nonlocal_3_idx = sorted(
            [self.layers[2] - (i + 1) for i in range(non_layers[2])])
        self.nonlocal_4 = nn.CellList(
            [NonLocal(2048) for i in range(non_layers[3])])
        self.nonlocal_4_idx = sorted(
            [self.layers[3] - (i + 1) for i in range(non_layers[3])])

        if non_layers[0] == 0:
            self.nonlocal_1_idx = [-1]
        if non_layers[1] == 0:
            self.nonlocal_2_idx = [-1]
        if non_layers[2] == 0:
            self.nonlocal_3_idx = [-1]
        if non_layers[3] == 0:
            self.nonlocal_4_idx = [-1]

        self.len1 = len(self.base_resnet.layer1)
        self.len2 = len(self.base_resnet.layer2)
        self.len3 = len(self.base_resnet.layer3)
        self.len4 = len(self.base_resnet.layer4)
        self.list1 = []
        self.list2 = []
        self.list3 = []
        self.list4 = []
        for item in self.base_resnet.layer1:
            self.list1.append(item)
        for item in self.base_resnet.layer2:
            self.list2.append(item)
        for item in self.base_resnet.layer3:
            self.list3.append(item)
        for item in self.base_resnet.layer4:
            self.list4.append(item)

    def construct(self, x):
        """
        non local output: used only non_local == "on"
        """

        x = self.base_resnet.conv1(x)
        x = self.base_resnet.bn1(x)
        # x = self.base_resnet.relu(x)
        x = self.base_resnet.maxpool(x)

        nonlocal1_counter = 0
        for i in range(self.len1):
            x = self.list1[i](x)
            if i == self.nonlocal_1_idx[nonlocal1_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_1[nonlocal1_counter](x)
                nonlocal1_counter += 1
        # Layer 2
        nonlocal2_counter = 0
        for i in range(self.len2):
            x = self.list2[i](x)
            if i == self.nonlocal_2_idx[nonlocal2_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_2[nonlocal2_counter](x)
                nonlocal2_counter += 1
        # Layer 3
        nonlocal3_counter = 0
        for i in range(self.len3):
            x = self.list3[i](x)
            if i == self.nonlocal_3_idx[nonlocal3_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_3[nonlocal3_counter](x)
                nonlocal3_counter += 1
        # Layer 4
        nonlocal4_counter = 0
        for i in range(self.len4):
            x = self.list4[i](x)
            if i == self.nonlocal_4_idx[nonlocal4_counter]:
                # _, C, H, W = x.shape
                x = self.nonlocal_4[nonlocal4_counter](x)
                nonlocal4_counter += 1
        return x


def resnet50nl(pretrain="", last_conv_stride=1):

    resnet_nl = ResNetNL(pretrain=pretrain, last_conv_stride=last_conv_stride)

    if pretrain:
        print('Non-Local resnet-50 pretrained weight loaded!')
        # loaded in ResNetNL.__init__(), where resnet50(pretrain, last_conv_stride) was called

    return resnet_nl
