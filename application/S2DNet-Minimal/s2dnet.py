from mindspore import nn
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal
from typing import List, Dict

from adap_layers import AdapLayers
from vgg16 import vgg16_layers

# VGG16架构定义
class VGG16(nn.Cell):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        # Feature Extraction
        self.features = nn.SequentialCell([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # Classification
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, 4096, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(4096, num_classes, weight_init=Normal(0.02)),
        ])

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.classifier(x)
        return x

class S2DNet(nn.Cell):
    """The S2DNet model
    """

    def __init__(
        self,
        hypercolumn_layers: List[str] = ["conv1_2", "conv3_3", "conv5_3"],
        checkpoint_path: str = None,
    ):
        """Initialize S2DNet.

        Args:
            hypercolumn_layers: Names of the layers to extract features from
            checkpoint_path: Path to the pre-trained model.
        """
        super(S2DNet, self).__init__()
        self._checkpoint_path = checkpoint_path
        self.layer_to_index = dict((k, v) for v, k in enumerate(vgg16_layers.keys()))
        self._hypercolumn_layers = hypercolumn_layers

        vgg16 = VGG16()
        layers = list(vgg16.features.cells())[:-2]
        self.encoder = nn.SequentialCell(*layers)
        
        self.adaptation_layers = AdapLayers(self._hypercolumn_layers)

        # Restore params from checkpoint
        if checkpoint_path:
            print(">> Loading weights from {}".format(checkpoint_path))
            param_dict = load_checkpoint(checkpoint_path)
            load_param_into_net(self, param_dict)
            if "hypercolumn_layers" in param_dict:
                self._hypercolumn_layers = param_dict["hypercolumn_layers"].asnumpy().tolist()

    def construct(self, image_tensor: Tensor):
        """Compute intermediate feature maps at the provided extraction levels.

        Args:
            image_tensor: The [N x 3 x H x Ws] input image tensor.
        Returns:
            feature_maps: The list of output feature maps.
        """
        feature_maps, j = [], 0
        feature_map = image_tensor
        layer_list = list(self.encoder.cells())
        for i, layer in enumerate(layer_list):
            feature_map = layer(feature_map)
            if j < len(self._hypercolumn_layers):
                next_extraction_index = self.layer_to_index[self._hypercolumn_layers[j]]
                if i == next_extraction_index:
                    feature_maps.append(feature_map)
                    j += 1
        feature_maps = self.adaptation_layers(feature_maps)
        return feature_maps
