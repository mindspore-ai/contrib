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
"""SpectralNorm"""

from mindspore.nn import Norm
from mindspore import numpy as ms_np
from mindspore import nn, Parameter
from mindspore._extends import cell_attr_register
import mindspore.nn.probability.distribution as msd

__all__ = ["SpectralNorm"]


def l2normalize(v, eps=1e-12):
    """l2normalize"""
    norm = Norm()
    return v / (norm(v) + eps)


class SpectralNorm(nn.Cell):
    """SpectralNorm"""

    @cell_attr_register
    def __init__(self, module, n_power_iterations=1, eps=1e-12):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        """_update_u_v"""
        u = self.module.weight_u
        v = self.module.weight_v
        w = self.module.weight_bar

        height = w.data.shape[0]
        for _ in range(self.n_power_iterations):
            v.set_data(l2normalize(ms_np.multi_dot([w.data.view(height, -1).transpose(), u.data]), eps=self.eps))
            u.set_data(l2normalize(ms_np.multi_dot([w.data.view(height, -1), v.data]), eps=self.eps))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = ms_np.dot(u, ms_np.multi_dot([w.data.view(height, -1), v.data]))
        self.module.weight = w / sigma.expand_as(w)

    def _made_params(self):
        try:
            u = self.module.weight_u
            v = self.module.weight_v
            w = self.module.weight_bar
            print(u, v, w)
            return True
        except KeyError:
            return False

    def _make_params(self):
        """make_params"""
        w = self.module.weight

        height = w.shape[0]
        width = w.view(height, -1).shape[1]

        uv = msd.Normal(0, 1, dtype=w.dtype)
        u_data = uv.prob(ms_np.arange(height))
        v_data = uv.prob(ms_np.arange(width))
        u_data = l2normalize(u_data, eps=self.eps)
        v_data = l2normalize(v_data, eps=self.eps)
        u = Parameter(u_data, name="weight_u", requires_grad=False)
        v = Parameter(v_data, name="weight_v", requires_grad=False)

        w_bar = Parameter(w, name="weight_bar")

        del self.module.weight
        self.module.insert_param_to_cell("weight_u", u)
        self.module.insert_param_to_cell("weight_v", v)
        self.module.insert_param_to_cell("weight_bar", w_bar)

    def construct(self, *args):
        self._update_u_v()
        return self.module.construct(*args)
