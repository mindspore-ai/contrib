import mindspore.nn as nn
from mindspore import ops
from mindvision.classification.models import MobileNetV2


class LightHead(nn.Cell):
    def __init__(self, in_, backbone, mode="S", out_size="Thin") -> None:
        """

        :param in_: output features of the backbone you try to light head
        :param backbone:
        :param mode:
        :param out_size:
        """
        super(LightHead, self).__init__()
        assert "S" in mode or "L" in mode, "Please specity the correct Light head mode"
        assert "Thin" in out_size or "Large" in out_size, "Please specify the model out size"
        self.backbone = backbone
        if mode == "L":
            self.out_mode = 256
        else:
            self.out_mode = 64


        if out_size =="Thin":
            self.c_out = 10
        else:
            self.c_out = 10 * 7 * 7

        self.conv1 = nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1, pad_mode='pad', padding=(7, 7, 0, 0), has_bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.out_mode, out_channels=self.c_out, kernel_size=(1, 15),  stride=1, pad_mode='pad', padding=(0, 0, 7, 7), has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1, pad_mode='pad', padding=(7, 7, 0, 0), has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=self.out_mode, out_channels=self.c_out, kernel_size=(1, 15), stride=1, pad_mode='pad', padding=(0, 0, 7, 7), has_bias=True)

    def construct(self, input):
        x_backbone = self.backbone(input)
        x = self.conv1(x_backbone)
        x = self.relu(x)
        x = self.conv2(x)
        x_relu_2 = self.relu(x)

        x = self.conv3(x_backbone)
        x = self.relu(x)
        x = self.conv4(x)
        x_relu_4 = self.relu(x)

        return x_relu_2 + x_relu_4


if __name__ == '__main__':
    backbone = MobileNetV2()
    backbone = LightHead(1280, backbone, mode="S", out_size="Thin")

    x = ops.randn([1, 3, 224, 224])
    output = backbone(x)
    print(output.shape)
    print(output)
