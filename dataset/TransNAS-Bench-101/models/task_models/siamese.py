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
"""siamese"""

import sys
from pathlib import Path

import mindspore.nn as nn
from mindspore.ops import operations as P

lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


class SiameseNet(nn.Cell):
    """SiameseNet used in Jigsaw task"""
    def __init__(self, encoder, decoder):
        super(SiameseNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, x):
        """construct"""
        if len(x.shape) == 4:
            assert x.shape == (1, 3, 720, 1080)
            x = image2tiles4testing(x)
        imgtile_num = x.shape[1]
        encoder_output = []
        for index in range(imgtile_num):
            input_i = x[:, index, :, :, :]
            ith_encoder_output = self.encoder(input_i)
            encoder_output.append(ith_encoder_output)
        concat_output = P.Concat()(encoder_output)
        final_output = self.decoder(concat_output)
        return final_output


def image2tiles4testing(img, num_pieces=9):
    """
    Generate the 9 pieces input for Jigsaw task.

    Parameters:
    -----------
        img (tensor): Image to be cropped (1, 3, 720, 1080)
h
    Return:
    -----------
        img_tiles: tensor (1, 9, 3, 240, 360)
    """

    if num_pieces != 9:
        raise ValueError(f'Target permutation of Jigsaw is supposed to have length 9, getting {num_pieces} here')

    ba, ch, he, wi = img.shape  # (1, 3, 720, 1080)

    unit_h = int(he / 3)  # 240
    unit_w = int(wi / 3)  # 360

    return img.view(ba, 9, ch, unit_h, unit_w)
