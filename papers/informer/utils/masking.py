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
"""masking"""

from mindspore import dtype
from mindspore import numpy as ms_np


class TriangularCausalMask():
    """TriangularCausalMask"""
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        self.mask = ms_np.triu(ms_np.ones(mask_shape).astype(dtype.bool_), k=1)

    @property
    def mask(self):
        return self.mask


class ProbMask():
    """ProbMask"""
    def __init__(self, B, H, L, index, scores):
        mask = ms_np.triu(ms_np.ones((L, scores.shape[-1])).astype(dtype.bool_), k=1)
        mask_ex = mask[None, None, :].expand_as(ms_np.ones((B, H, L, scores.shape[-1])))
        indicator = mask_ex[ms_np.arange(B)[:, None, None],
                            ms_np.arange(H)[None, :, None],
                            index, :]
        self.mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self.mask
