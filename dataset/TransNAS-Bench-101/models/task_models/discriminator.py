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
"""discriminator"""

import sys
from pathlib import Path

import mindspore.nn as nn
import mindspore.ops.operations as P

from net_ops.cell_ops import ConvLayer
from net_ops.spectral_norm import SpectralNorm

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


class Discriminator(nn.Cell):
    """Discriminator"""

    def __init__(self, norm="spectral"):
        """
        Discriminator component for Pix2Pix tasks
        :param norm: ["batch": BN, "spectral": spectral norm for GAN]
        """
        super(Discriminator, self).__init__()
        if norm == "batch":
            norm = nn.BatchNorm2d
        elif norm == "spectral":
            norm = SpectralNorm
        else:
            raise ValueError(f"{norm} is invalid!")
        self.avgpool = P.ReduceMean()

        # input: [batch x 6 x 256 x 256]
        self.conv1 = ConvLayer(6, 64, 5, 4, 2, nn.LeakyReLU(alpha=0.2), norm)
        self.conv2 = ConvLayer(64, 128, 5, 4, 2, nn.LeakyReLU(alpha=0.2), norm)
        self.conv3 = ConvLayer(128, 256, 5, 4, 2, nn.LeakyReLU(alpha=0.2), norm)
        self.conv4 = ConvLayer(256, 256, 3, 1, 1, nn.LeakyReLU(alpha=0.2), norm)
        self.conv5 = ConvLayer(256, 512, 3, 1, 1, nn.LeakyReLU(alpha=0.2), norm)
        self.conv6 = ConvLayer(512, 512, 3, 1, 1, nn.LeakyReLU(alpha=0.2), norm)
        self.conv7 = ConvLayer(512, 1, 3, 1, 1, None, None)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x, (1, 1))
        return P.Flatten()(x)
