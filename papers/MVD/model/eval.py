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
"""eval"""

import time
import os
import numpy as np
import psutil


def show_memory_info(hint=""):
    """show_memory_info"""
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")


def test(args, gallery, query, ngall, nquery,
         backbone, gall_modal, gallery_cam=None, query_cam=None):
    """test"""
    print('Extracting Gallery Feature...', gall_modal)
    start = time.time()
    ptr = 0

    gall_feat_v = np.zeros((ngall, 2048 + args.z_dim))
    gall_feat_i = np.zeros((ngall, 2048 + args.z_dim))
    gall_feat_v_shared = np.zeros((ngall, 2048 + args.z_dim))
    gall_feat_i_shared = np.zeros((ngall, 2048 + args.z_dim))
    gall_label = np.zeros((ngall,))
    query_label = np.zeros((nquery,))
    for (img, label) in gallery:
        feat_v, feat_v_shared, feat_i, feat_i_shared = backbone(img, mode=1)
        size = int(feat_v.shape[0])
        gall_feat_v[ptr:ptr + size, :] = feat_v.asnumpy()
        gall_feat_v_shared[ptr:ptr + size, :] = feat_v_shared.asnumpy()
        gall_feat_i[ptr:ptr + size, :] = feat_i.asnumpy()
        gall_feat_i_shared[ptr:ptr + size, :] = feat_i_shared.asnumpy()

        gall_label[ptr:ptr + size] = label.asnumpy()

        ptr = ptr + size
    print('Extracting Time x4:\t {:.3f}'.format(time.time() - start))
    # print("gallery label:", gall_label)

    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat_v = np.zeros((nquery, 2048 + args.z_dim))
    query_feat_i = np.zeros((nquery, 2048 + args.z_dim))
    query_feat_v_shared = np.zeros((nquery, 2048 + args.z_dim))
    query_feat_i_shared = np.zeros((nquery, 2048 + args.z_dim))
    for (img, label) in query:
        feat_v, feat_v_shared, feat_i, feat_i_shared = backbone(img, mode=1)
        size = int(feat_v.shape[0])
        query_feat_v[ptr:ptr + size, :] = feat_v.asnumpy()
        query_feat_v_shared[ptr:ptr + size, :] = feat_v_shared.asnumpy()
        query_feat_i[ptr:ptr + size, :] = feat_i.asnumpy()
        query_feat_i_shared[ptr:ptr + size, :] = feat_i_shared.asnumpy()

        query_label[ptr:ptr + size] = label.asnumpy()

        ptr = ptr + size
    print('Extracting Time x4:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    print(start)
    # compute the similarity
    distmat_v = np.matmul(query_feat_v, np.transpose(gall_feat_v))
    distmat_v_shared = np.matmul(query_feat_v_shared, np.transpose(gall_feat_v_shared))
    distmat_i = np.matmul(query_feat_i, np.transpose(gall_feat_i))
    distmat_i_shared = np.matmul(query_feat_i_shared, np.transpose(gall_feat_i_shared))

    cmc_v = map_v = cmc_i = map_i = None
    if args.dataset == "SYSU":
        cmc1, map1 = eval_sysu(-distmat_v, query_label, gall_label, query_cam, gallery_cam)
        cmc2, map2 = eval_sysu(-distmat_v_shared, query_label, gall_label, query_cam, gallery_cam)
        cmc3, map3 = eval_sysu(-distmat_i, query_label, gall_label, query_cam, gallery_cam)
        cmc4, map4 = eval_sysu(-distmat_i_shared, query_label, gall_label, query_cam, gallery_cam)

        cmc_v = (cmc1 + cmc2) / 2.0
        map_v = (map1 + map2) / 2.0
        cmc_i = (cmc3 + cmc4) / 2.0
        map_i = (map3 + map4) / 2.0

    # elif args.dataset == "RegDB":
    #     cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
    #     cmc_att, mAP_att = eval_regdb(-distmat_att, query_label, gall_label)

    return cmc_v, map_v, cmc_i, map_i


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # reference: Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_inp.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC
    print(all_cmc)

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    maps = np.mean(all_ap)
    minp = np.mean(all_inp)
    print(minp)
    return new_all_cmc, maps


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """eval_regdb"""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_ap = []
    all_inp = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # reference: Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_inp.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    maps = np.mean(all_ap)
    minp = np.mean(all_inp)
    print(minp)
    return all_cmc, maps
