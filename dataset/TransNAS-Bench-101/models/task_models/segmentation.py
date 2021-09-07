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
"""Segmentation"""

import sys
from pathlib import Path

import mindspore.nn as nn

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


class Segmentation(nn.Cell):
    """Segmentation used by semantic segment task"""

    def __init__(self, encoder, decoder):
        super(Segmentation, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, x):
        return self.decoder(self.encoder(x))
