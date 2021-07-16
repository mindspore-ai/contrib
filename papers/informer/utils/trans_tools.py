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
"""trans_tools"""

from mindspore.ops import Select
from mindspore import numpy as ms_np
from mindspore import dtype


def trans_shape(trans_index, tensor_shape):
    """trans_shape"""
    s = list(tensor_shape)
    s[trans_index[0]], s[trans_index[1]] = s[trans_index[1]], s[trans_index[0]]
    return tuple(s)


def mask_fill(mask, data, num):
    """mask_fill"""
    select = Select()
    if not mask.shape == data.shape:
        mask = mask.expand_as(data)
    replace_tensor = ms_np.ones(data.shape)
    replace_tensor[:] = num
    return select(mask, data.astype(dtype.float32), replace_tensor)
