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
"""attn"""

import math
from math import sqrt

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import Softmax, ExpandDims, UniformInt, TopK
from mindspore import Tensor, dtype
from mindspore import numpy as ms_np

from utils.masking import TriangularCausalMask, ProbMask
from utils.trans_tools import trans_shape, mask_fill


class FullAttention(nn.Cell):
    """FullAttention"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(keep_prob=1.0 - attention_dropout)
        self.softmax = Softmax()
        self.factor = factor

    def construct(self, queries, keys, values, attn_mask):
        """FullAttention"""
        b, l, _, e = queries.shape
        scale = self.scale or 1. / sqrt(e)

        scores = ms_np.matmul(queries.transpose((0, 2, 1, 3)), keys.transpose((0, 2, 3, 1)))

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(b, l)

            mask_fill(attn_mask.mask, scores, -math.inf)

        a = self.dropout(self.softmax(Tensor(scale * scores, dtype.float32)))
        v = ms_np.matmul(a, values.transpose((0, 2, 1, 3)))
        v = v.transpose(trans_shape((1, 2), v.shape))

        if self.output_attention:
            return(v, a)
        return(v, None)


class ProbAttention(nn.Cell):
    """ProbAttention"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(keep_prob=1.0 - attention_dropout)
        self.expand = ExpandDims()
        self.uniform_int = UniformInt()
        self.softmax = Softmax()
        self.topk = TopK()

    def _prob_qk(self, q, k, sample_k, n_top):  # n_top: c*ln(L_q)
        """_prob_qk"""
        # q [B, H, L, ]
        b, h, l_k, e = k.shape
        _, _, l_q, _ = q.shape

        # calculate the sampled Q_K
        k_expand = self.expand(k, -3).expand_as(ms_np.ones((b, h, l_q, l_k, e)))
        index_sample = self.uniform_int((l_q, sample_k), 0, l_k)  # real U = U_part(factor*ln(L_k))*L_q
        k_sample = k_expand[:, :, self.expand(ms_np.arange(l_q), 1), index_sample, :]
        q_k_sample = ms_np.matmul(self.expand(q, -2),
                                  k_sample.transpose(trans_shape((-1, -2), k_sample.shape))).squeeze()

        # find the Top_k query with sparisty measurement
        m = q_k_sample.max(-1)[0] - P.Div()(q_k_sample.sum(-1), l_k)
        m_top = self.topk(m, n_top)[1]

        # use the reduced q to calculate Q_K
        q_reduce = q[ms_np.arange(b)[:, None, None],
                     ms_np.arange(h)[None, :, None],
                     m_top, :]  # factor*ln(L_q)
        q_k = ms_np.matmul(q_reduce, k.transpose(trans_shape((-1, -2), k.shape)))  # factor*ln(L_q)*L_k

        return q_k, m_top

    def _get_initial_context(self, v, l_q):
        b, h, l_v, _ = v.shape
        if not self.mask_flag:
            # V_sum = v.sum(dim=-2)
            v_sum = P.ReduceMean()(v, -2)
            context = self.expand(v_sum, -2).expand_as(ms_np.ones((b, h, l_q, v_sum.shape[-1]))).copy()
        else:  # use mask
            assert l_q == l_v  # requires that l_q == l_v, i.e. for self-attention only
            context = v.cumsum(axis=-2)
        return context

    def _update_context(self, context_in, v, scores, index, l_q):
        """_update_context"""
        b, h, l_v, _ = v.shape

        if self.mask_flag:
            attn_mask = ProbMask(b, h, l_q, index, scores)
            scores.masked_fill_(attn_mask.mask, -math.inf)

        attn = self.softmax(scores)

        context_in[ms_np.arange(b)[:, None, None],
                   ms_np.arange(h)[None, :, None],
                   index, :] = ms_np.matmul(attn, v).astype(context_in.dtype)
        if self.output_attention:
            attns = (ms_np.ones([b, h, l_v, l_v]) / l_v).astype(attn.dtype)  # ?#.to(attn.device)
            attns[ms_np.arange(b)[:, None, None], ms_np.arange(h)[None, :, None], index, :] = attn
            return(context_in, attns)
        return(context_in, None)

    def construct(self, queries, keys, values):
        """build"""
        _, l_q, _, d = queries.shape
        _, l_k, _, _ = keys.shape

        queries = queries.transpose(trans_shape((1, 2), queries.shape))
        keys = keys.transpose(trans_shape((1, 2), keys.shape))
        values = values.transpose(trans_shape((1, 2), values.shape))

        u_part = self.factor * math.ceil(math.log(l_k))  # c*ln(L_k)
        u = self.factor * math.ceil(math.log(l_q))  # c*ln(L_q)

        u_part = u_part if u_part < l_k else l_k
        u = u if u < l_q else l_q

        scores_top, index = self._prob_QK(queries, keys, sample_k=u_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(d)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, l_q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, l_q)

        return context.transpose(trans_shape((1, 2), context.shape)), attn


class AttentionLayer(nn.Cell):
    """AttentionLayer"""
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(in_channels=d_model, out_channels=d_keys * n_heads)
        self.key_projection = nn.Dense(in_channels=d_model, out_channels=d_keys * n_heads)
        self.value_projection = nn.Dense(in_channels=d_model, out_channels=d_values * n_heads)
        self.out_projection = nn.Dense(in_channels=d_values * n_heads, out_channels=d_model)
        self.n_heads = n_heads
        self.mix = mix

    def construct(self, queries, keys, values, attn_mask):
        """build"""
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(trans_shape((1, 2), out.shape))
        out = P.Reshape()(out, (b, l, -1,))

        return self.out_projection(out), attn
