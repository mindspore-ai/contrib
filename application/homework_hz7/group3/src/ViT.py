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
"""Vision Model."""
import mindspore.nn as nn
from mindvision.classification.models.backbones import ViT


class VisionModel(nn.Cell):
    """
    Vision Transformer architecture implementation.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, 768)`

    Raises:
        ValueError: If `split` is not 'train', "test or 'infer'.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindvision.classification.models.backbones import ViT
        >>> net = ViT()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 768)

    About ViT:

    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.

    Citation:

    .. code-block::

        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """


    def __init__(self):
        super().__init__()
        self.vit = ViT(image_size=512, pool='cls')

    def construct(self, x):
        """
        1
        :param x:1
        :return:
        """
        return self.vit(x)
