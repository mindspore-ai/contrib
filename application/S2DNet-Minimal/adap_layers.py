import mindspore.nn as nn
from mindspore import Tensor
from typing import List

from vgg16 import vgg16_layers

class AdapLayers(nn.Cell):
    """Small adaptation layers.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdapLayers, self).__init__()
        self.layers = []
        channel_sizes = [vgg16_layers[name] for name in hypercolumn_layers]
        for i, channels in enumerate(channel_sizes):
            layer = nn.SequentialCell([
                nn.Conv2d(channels, 64, kernel_size=1, stride=1, pad_mode='pad', has_bias=True, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, output_dim, kernel_size=5, stride=1, pad_mode='pad', has_bias=True, padding=2),
                nn.BatchNorm2d(output_dim),
            ])
            self.layers.append(layer)
            self.insert_child_to_cell("adap_layer_{}".format(i), layer)  # 在MindSpore中添加子模块

    def construct(self, features: List[Tensor]):
        """Apply adaptation layers.
        """
        for i, _ in enumerate(features):
            features[i] = getattr(self, "adap_layer_{}".format(i))(features[i])
        return features
