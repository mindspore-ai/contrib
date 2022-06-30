"""ResNet"""
from mindvision.classification.models.backbones import ResidualBlockBase, ResNet
from mindvision.classification.models.resnet import _resnet


def ResNet10(num_classes: int = 10,
             pretrained: bool = False
             ) -> ResNet:
    return _resnet(
        "resnet10", ResidualBlockBase, [
            1, 1, 1, 1], num_classes, pretrained, 512)


def ResNet12(num_classes: int = 10,
             pretrained: bool = False
             ) -> ResNet:
    return _resnet(
        "resnet12", ResidualBlockBase, [
            2, 1, 1, 1], num_classes, pretrained, 512)


def ResNet18(num_classes: int = 10,
             pretrained: bool = False,
             ):
    return _resnet(
        "resnet18", ResidualBlockBase, [
            2, 2, 2, 2], num_classes, pretrained, 512)
