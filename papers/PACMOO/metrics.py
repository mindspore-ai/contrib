"""
This data implement model's metric

:paper author: hxq
:code author: hxq
:code convert: shy
"""
import math
from sklearn import metrics
import numpy as np
import bottleneck as bn


def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1) # shape和x_pred一样
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]] # 所有用户的topk item同时挑出来
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tset_pre = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tset_pre).sum(axis=1)
    idcg = np.array([(tset_pre[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def precision_recall_at_k_batch(x_pred, heldout_batch, k=100,
                                observe_fair=False, attr_indicator_list=None):
    """[summary]

    Args:
        x_pred ([type]): [description]
        heldout_batch ([type]): [description]
        k (int, optional): [description]. Defaults to 100.
        observe_fair (bool, optional): [description]. Defaults to False.
        attr_indicator_list ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    print(observe_fair, attr_indicator_list)
    batch_users = x_pred.shape[0]

    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()

    # ranked_list = x_pred_binary

    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / x_true_binary.sum(axis=1)
    precision = tmp / k
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    f1_recall = 2 * recall * precision / (precision + recall)
    f1_recall[np.isnan(f1_recall)] = 0
    return precision, recall, f1_recall


def update_threshold(x_pred, id_onehots_ph, threshold_ph, k=100):
    """[summary]

    Args:
        x_pred ([type]): [description]
        id_onehots_ph ([type]): [description]
        threshold_ph ([type]): [description]
        k (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    batch_users = x_pred.shape[0]
    idx = bn.argpartition(-x_pred, k, axis=1)
    #epsion = 1e-10
    #threshold_ph_batch = x_pred[:, idx[:, k]]-epsion
    #print('shape(threshold_ph_batch)', threshold_ph_batch.shape)
    threshold_ph[np.nonzero(id_onehots_ph)[1]] = \
    x_pred[np.arange(batch_users), idx[:, k]].reshape(-1, 1)
    #threshold_ph = np.dot(threshold_ph.T, id_onehots_ph.toarray())
    return threshold_ph


def average_precision(ranked_list, ground_truth):
    """Compute the average precision (AP) of a list of ranked items
    """
    hits = 0
    sum_precs = 0
    for index in range(len(ranked_list)):
        if ranked_list[index] in ground_truth:
            hits += 1
            sum_precs += hits / (index + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    return 0


def hit(gt_items, pred_items):  # HR为所有用户的hits/所有用户的grounf truth总个数
    """[summary]

    Args:
        gt_items ([type]): [description]
        pred_items ([type]): [description]

    Returns:
        [type]: [description]
    """
    count = 0
    for item in pred_items:
        if item in gt_items:
            count += 1
    return count


def auc(label, prob): # prob 为预测为正的概率
    """[summary]

    Args:
        label ([type]): [description]
        prob ([type]): [description]

    Returns:
        [type]: [description]
    """
    precision, recall, thresholds = metrics.precision_recall_curve(label, prob)
    print(thresholds)
    area = metrics.auc(recall, precision)
    return area
# sklearn
# precision, recall, _thresholds = metrics.precision_recall_curve(label, prob)
# area = metrics.auc(recall, precision)
# return area


# area = metrics.roc_auc_score(label, prob)
# return area

def hit_precision_recall_ndcg_k(train_set_batch, test_set_batch,
                                pred_scores_batch, max_train_count,
                                k=20, ranked_tag=False, vad_set_batch=None):
    """[summary]

    Args:
        train_set_batch ([type]): [description]
        test_set_batch ([type]): [description]
        pred_scores_batch ([type]): [description]
        max_train_count ([type]): [description]
        k (int, optional): [description]. Defaults to 20.
        ranked_tag (bool, optional): [description]. Defaults to False.
        vad_set_batch ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    recall_k, precision_k, ndcg_k, hits_list = [], [], [], []

    if not ranked_tag:
        batch_users = pred_scores_batch.shape[0]
        idx_topk_part = bn.argpartition(-pred_scores_batch, k+max_train_count, axis=1)
        topk_part = pred_scores_batch[np.arange(batch_users)[:, np.newaxis],
                                      idx_topk_part[:, :(k+max_train_count)]]
        idx_part = np.argsort(-topk_part, axis=1)
        top_items = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    else:
        top_items = pred_scores_batch
    if vad_set_batch is None:
        for train_set, test_set, ranked in zip(train_set_batch, test_set_batch, top_items):

            n_k = k if len(test_set) > k else len(test_set) # n_k = min(k, len(test_k))

            n_idcg, n_dcg = 0, 0
            for pos in range(n_k):
                n_idcg += 1.0 / math.log(pos + 2, 2)

            tops_sub_train = []
            n_top_items = 0
            for val in ranked:
                if val not in train_set:
                    tops_sub_train.append(val)

                n_top_items += 1

                if n_top_items >= k: # 控制topK个item是从用户没交互过的商品中选的
                    break
            hits_set = [(idx, item_id) for idx, item_id in enumerate(tops_sub_train) if item_id in test_set]
            cnt_hits = len(hits_set)

            for idx in range(cnt_hits):
                n_dcg += 1.0 / math.log(hits_set[idx][0] + 2, 2)
            precision_k.append(float(cnt_hits / k))
            recall_k.append(float(cnt_hits / len(test_set)))
            ndcg_k.append(float(n_dcg / n_idcg))
            hits_list.append(cnt_hits)
    else:
        hits_list, precision_k, recall_k, ndcg_k = \
            calc_second(train_set_batch, test_set_batch, top_items, vad_set_batch, k)
    return hits_list, precision_k, recall_k, ndcg_k


def calc_second(train_set_batch, test_set_batch, top_items, vad_set_batch, k):
    """[summary]

    Args:
        train_set_batch ([type]): [description]
        test_set_batch ([type]): [description]
        top_items ([type]): [description]
        vad_set_batch ([type]): [description]
        k ([type]): [description]

    Returns:
        [type]: [description]
    """
    recall_k, precision_k, ndcg_k, hits_list = [], [], [], []
    for train_set, test_set, ranked, vad_set in \
        zip(train_set_batch, test_set_batch, top_items, vad_set_batch):

        n_k = k if len(test_set) > k else len(test_set) # n_k = min(k, len(test_k))

        n_idcg, n_dcg = 0, 0
        for pos in range(n_k):
            n_idcg += 1.0 / math.log(pos + 2, 2)

        tops_sub_train = []
        n_top_items = 0
        for val in ranked:
            if val not in train_set and val not in vad_set:
                tops_sub_train.append(val)

            n_top_items += 1

            if n_top_items >= k: # 控制topK个item是从用户没交互过的商品中选的
                break
        hits_set = [(idx, item_id) for idx, item_id in \
                    enumerate(tops_sub_train) if item_id in test_set]
        cnt_hits = len(hits_set)

        for idx in range(cnt_hits):
            n_dcg += 1.0 /math.log(hits_set[idx][0] + 2, 2)
        precision_k.append(float(cnt_hits / k))
        recall_k.append(float(cnt_hits / len(test_set)))
        ndcg_k.append(float(n_dcg / n_idcg))
        hits_list.append(cnt_hits)

    return hits_list, precision_k, recall_k, ndcg_k
