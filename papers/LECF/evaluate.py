# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
eval.py
"""
import numpy as np


def leave_one_out(purchased_item, recommend_list, top_k_recommand_number):
    '''
    get metrics
    '''
    top_recommend_list = recommend_list[:top_k_recommand_number]
    if purchased_item in top_recommend_list:
        return 1, np.log2(2.0) / np.log2(top_recommend_list.index(purchased_item) + 2.0)

    return 0, 0


def ndcg_k(recommend_list, purchased_list):
    '''
    NDCG
    '''
    z_u = 0
    temp = 0
    for j in range(min(len(recommend_list), len(purchased_list))):
        z_u = z_u + 1 / np.log2(j + 2)
    for j in range(len(recommend_list)):
        if recommend_list[j] in purchased_list:
            temp = temp + 1 / np.log2(j + 2)
    if z_u == 0:
        temp = 0
    else:
        temp = temp / z_u
    return temp


def top_k(recommend_list, purchased_list):
    '''
    HR
    '''
    temp = []
    hr = 0
    for j in recommend_list:
        if j in purchased_list:
            temp.append(j)
            hr = 1

    co_length = len(temp)
    re_length = len(recommend_list)
    pu_length = len(purchased_list)

    if re_length == 0:
        p = 0.0
    else:
        p = co_length / float(re_length)

    if pu_length == 0:
        r = 0.0
    else:
        r = co_length / float(pu_length)

    if r != 0 or p != 0:
        f = 2.0 * p * r / (p + r)
    else:
        f = 0.0
    return p, r, f, hr
