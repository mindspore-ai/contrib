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
"""encoder"""

import sys
from pathlib import Path

import mindspore.nn as nn

from net_infer.net_macro import MacroNet
from net_ops.resnet import resnet50, ResidualBlock

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


class FFEncoder(nn.Cell):
    """Encoder class for the definition of backbone including resnet50 and MacroNet()"""

    def __init__(self, encoder_str, task_name=None):
        super(FFEncoder, self).__init__()
        self.encoder_str = encoder_str

        # Initialize network
        if self.encoder_str == 'resnet50':
            self.network = resnet50()  # resnet50: Bottleneck, [3,4,6,3]
            # Adjust according to task
            if task_name in ['autoencoder', 'normal', 'inpainting', 'segmentsemantic']:
                self.network.layer4 = self.network.make_layer(
                    ResidualBlock, 3, 1024, 512, 1)
                self.network = nn.SequentialCell([
                    *list(self.network.cells())[:-2],
                ])
            else:
                self.network = nn.SequentialCell([*list(self.network.cells())[:-2]])
        else:
            self.network = MacroNet(encoder_str, structure='backbone')

    def construct(self, x):
        x = self.network(x)
        return x
