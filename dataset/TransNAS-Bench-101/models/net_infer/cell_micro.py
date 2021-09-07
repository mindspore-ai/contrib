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
"""cell_micro"""

import sys
from pathlib import Path

from mindspore import nn
from mindspore.ops import Stack

from net_ops.cell_ops import ReLUConvBN, OPS

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


class MicroCell(nn.Cell):
    """MicroCell"""
    expansion = 1

    def __init__(self, cell_code, c_in, c_out, stride, affine=True, track_running_stats=True):
        """
        initialize a cell
        Args:
            cell_code: ['', '1', '13', '302']
            c_in: in channel
            c_out: out channel
            stride: 1 or 2
        """
        super(MicroCell, self).__init__()

        if cell_code == "basic":
            raise "invalid {:} cell code".format(cell_code)

        if stride not in [1, 2]:
            raise "invalid stride {:}".format(stride)

        self.node_num = len(cell_code)
        self.edges = nn.CellList()
        self.nodes = list(range(len(cell_code)))  # e.g. [0, 1, 2, 3]
        if self.nodes != list(map(len, cell_code)):
            raise "invalid node {:}".format(self.node)
        self.from_nodes = [list(range(i)) for i in self.nodes]  # e.g. [[], [0], [0, 1], [0, 1, 2]]
        self.from_ops = [list(range(n * (n - 1) // 2, n * (n - 1) // 2 + n))
                         for n in range(self.node_num)]  # e.g. [[], [0], [1, 2], [3, 4, 5]]
        self.stride = stride

        for node in self.nodes:
            for op_idx, from_node in zip(cell_code[node], self.from_nodes[node]):
                if from_node == 0:
                    edge = OPS[op_idx](c_in, c_out, self.stride, affine, track_running_stats)
                else:
                    edge = OPS[op_idx](c_out, c_out, 1, affine, track_running_stats)
                self.edges.append(edge)

        self.cell_code = cell_code
        self.c_in = c_in
        self.c_out = c_out
        self.stack = Stack()

    def construct(self, inputs):
        """construct"""
        node_features = [inputs]

        # compute the out features for each nodes
        for node_idx in self.nodes:
            if node_idx == 0:
                continue
            node_feature_list = [self.edges[from_op](node_features[from_node]) for from_op, from_node in
                                 zip(self.from_ops[node_idx], self.from_nodes[node_idx])]
            # for i, nf in enumerate(node_feature_list):
            #     print(node_idx, self.from_nodes[node_idx][i], self.cell_code[node_idx][i], nf.shape)
            node_feature = self.stack(node_feature_list).sum(0)
            # print(f"node_idx {node_idx} output:{node_feature.shape}\n")
            node_features.append(node_feature)
        return node_features[-1]


class ResNetBasicblock(nn.Cell):
    """ResNetBasicblock"""
    expansion = 1

    def __init__(self, cell_code, inplanes, planes, stride=1, affine=True, track_running_stats=None, activation='relu'):
        super(ResNetBasicblock, self).__init__()
        if cell_code != 'basic':
            raise "invalid cel code {:}".format(cell_code)
        if stride not in [1, 2]:
            raise 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine, track_running_stats, activation=activation)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine, track_running_stats, activation=activation)
        self.downsample = None

        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.SequentialCell(
                [nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, has_bias=False,
                           pad_mode="pad"),
                 nn.BatchNorm2d(planes * self.expansion, affine, use_batch_statistics=track_running_stats)]
            )

    def construct(self, inputs):

        feature = self.conv_a(inputs)
        feature = self.conv_b(feature)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + feature
