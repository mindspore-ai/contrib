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
"""net_macro"""

import sys
from pathlib import Path

from mindspore import nn
from mindspore.ops import ReduceMean
from mindspore.common.initializer import Constant

from net_infer.cell_micro import ResNetBasicblock, MicroCell
from net_ops.cell_ops import ReLUConvBN
from net_ops.he_normal import HeNormal

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


class MacroNet(nn.Cell):
    """Adapted from torchvision/models/resnet.py"""

    def __init__(self, net_code, structure='full', input_dim=(224, 224), num_classes=75):
        super(MacroNet, self).__init__()
        if structure not in ['full', 'drop_last', 'backbone']:
            raise 'unknown structrue: %s' % repr(structure)
        self.structure = structure
        self._read_net_code(net_code)
        self.inplanes = self.base_channel
        self.feature_dim = [input_dim[0] // 4, input_dim[1] // 4]

        self.stem = nn.SequentialCell(
            [nn.Conv2d(3, self.base_channel // 2, kernel_size=3, stride=2, padding=1, dilation=1, has_bias=False,
                       pad_mode="pad"),
             nn.BatchNorm2d(self.base_channel // 2, affine=True, use_batch_statistics=False),
             ReLUConvBN(self.base_channel // 2, self.base_channel, kernel_size=3, stride=2,
                        padding=1, dilation=1, affine=True, track_running_stats=None)]
        )

        self.layers = []
        for i, layer_type in enumerate(self.macro_code):
            layer_type = int(layer_type)  # channel change: [2, 4]; stride change: [3, 4]
            target_channel = self.inplanes * 2 if layer_type % 2 == 0 else self.inplanes
            stride = 2 if layer_type > 2 else 1
            self.feature_dim = [self.feature_dim[0] // stride, self.feature_dim[1] // stride]
            layer = self._make_layer(self.cell, target_channel, 2, stride, True, None)
            self.insert_child_to_cell(f"layer{i}", layer)
            self.layers.append(f"layer{i}")

        self.avgpool = ReduceMean() if structure in ['drop_last', 'full'] else None
        self.head = nn.Dense(self.inplanes, num_classes) if structure in ['full'] else None
        self.henormal = HeNormal(mode='fan_out', nonlinearity='relu')
        self.constant0 = Constant(0)
        self.constant1 = Constant(1)

        if structure == 'full':
            self.output_dim = (1, num_classes)
        elif structure == 'drop_last':
            self.output_dim = (self.inplanes, 1, 1)
        elif structure == 'backbone':
            self.output_dim = (self.inplanes, *self.feature_dim)
        else:
            raise ValueError

        self._kaiming_init()

    def construct(self, x):
        """construct"""
        x = self.stem(x)

        for _, layer_name in enumerate(self.layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

        if self.structure in ['full', 'drop_last']:
            x = self.avgpool(x, (1, 1))
            x = x.view(x.size(0), -1)

        if self.structure == 'full':
            x = self.head(x)

        return x

    def _make_layer(self, cell, planes, num_blocks, stride=1, affine=True, track_running_stats=None):
        layers = [cell(self.micro_code, self.inplanes, planes, stride, affine, track_running_stats)]
        self.inplanes = planes * cell.expansion
        for _ in range(1, num_blocks):
            layers.append(
                cell(self.micro_code, self.inplanes, planes, 1, affine, track_running_stats=track_running_stats))
        return nn.SequentialCell(layers)

    def _read_net_code(self, net_code):
        net_code_list = net_code.split('-')
        self.base_channel = int(net_code_list[0])
        self.macro_code = net_code_list[1]
        if net_code_list[-1] == 'basic':
            self.micro_code = 'basic'
            self.cell = ResNetBasicblock
        else:
            self.micro_code = [''] + net_code_list[2].split('_')
            self.cell = MicroCell

    def _kaiming_init(self):
        # kaiming initialization
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                self.henormal(m.weight.data)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                self.constant1(m.weight)
                self.constant0(m.bias)
