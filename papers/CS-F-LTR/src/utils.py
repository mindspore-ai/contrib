"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def dcg(predicted_order):
    """[summary]

    Args:
        predicted_order ([type]): [description]

    Returns:
        [type]: [description]
    """
    i = 1
    cumulative_dcg = 0
    for x in predicted_order:
        cumulative_dcg += (2**x - 1) / (np.log(1 + i))
        i += 1
    return cumulative_dcg


def ndcg(predicted_order, k_pos=-1):
    """[summary]

    Args:
        predicted_order ([type]): [description]
        k_pos (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order[:min(k_pos, len(
        predicted_order))]) if k_pos != -1 else dcg(predicted_order)
    if our_dcg == 0:
        return 0
    max_dcg = dcg(sorted_list[:min(k_pos, len(predicted_order))]
                  ) if k_pos != -1 else dcg(sorted_list)
    ndcg_output = our_dcg / max_dcg
    return ndcg_output


def ndcg_lambdarank(predicted_order):
    """[summary]

    Args:
        predicted_order ([type]): [description]

    Returns:
        [type]: [description]
    """
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order)
    max_dcg = dcg(sorted_list)
    ndcg_output = our_dcg / max_dcg
    return ndcg_output


def delta_ndcg(order1, pos1, pos2):
    """[summary]

    Args:
        order1 ([type]): [description]
        pos1 ([type]): [description]
        pos2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    ndcg1 = ndcg_lambdarank(order1)
    order1[[pos2, pos1]] = order1[[pos1, pos2]]
    ndcg2 = ndcg_lambdarank(order1)
    return np.absolute(ndcg1 - ndcg2)


def predict_order_for_one_query(qid, query_indices, pred_all, pred_part, relevance_labels):
    """[summary]

    Args:
        qid ([type]): [description]
        query_indices ([type]): [description]
        pred_all ([type]): [description]
        pred_part ([type]): [description]
        relevance_labels ([type]): [description]

    Returns:
        [type]: [description]
    """
    _ = qid
    # 取出给定qid的所有查询
    if pred_part is not None:
        predicted_score = pred_part
    else:
        predicted_score = pred_all[query_indices]
    # print("C", predicted_score)
    # 把预测分数和查询id组成对
    pred_query_type = np.dtype(
        [('predicted_scores', predicted_score.dtype),
         ('query_int', query_indices.dtype)])
    pred_query = np.empty(len(predicted_score), dtype=pred_query_type)
    pred_query['predicted_scores'] = np.reshape(predicted_score, [-1])
    pred_query['query_int'] = query_indices
    # 按照分数从高到低排序
    scored_pred_query = np.sort(pred_query, order='predicted_scores')[::-1]
    # print("D", scored_pred_query)
    return relevance_labels[scored_pred_query['query_int']]


def calc_err(predicted_order):
    """[summary]

    Args:
        predicted_order ([type]): [description]

    Returns:
        [type]: [description]
    """
    # expected reciprocal rank
    err = 0
    prev_one_min_rel_prod = 1
    previous_rel = 0
    t = len(predicted_order) if len(predicted_order) < 10 else 10
    for r in range(t):
        rel_r = calc_ri(predicted_order, r)
        one_min_rel_prod = (1 - previous_rel) * prev_one_min_rel_prod
        err += (1 / (r + 1)) * rel_r * one_min_rel_prod
        prev_one_min_rel_prod = one_min_rel_prod
        previous_rel = rel_r

    return err


def calc_ri(predicted_order, i):
    """[summary]

    Args:
        predicted_order ([type]): [description]
        i ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (2 ** predicted_order[i] - 1) / (2 ** np.max(predicted_order))


def cal_map(predicted_order):
    """[summary]

    Args:
        predicted_order ([type]): [description]

    Returns:
        [type]: [description]
    """
    p_at_k = list()
    p_at_k.append(predicted_order[0])
    for i in range(1, len(predicted_order)):
        p_at_k.append((p_at_k[-1] * i + predicted_order[i]) / (i + 1))
    nomi = 0
    for i in range(len(predicted_order)):
        nomi += predicted_order[i] * p_at_k[i]
    nomi /= len(predicted_order)
    return nomi

# def cal_map(predicted_order):
#     n1 = 0
#     n2 = 0
#     for each in predicted_order:
#         if 0.2 < each < 1.2:
#             n1 += 1
#         elif each > 1.5:
#             n2 += 1
#     n = n1 + n2
#     if n == 0:
#         n = 1
#     cnt1 = 0
#     cnt2 = 0
#     ap = 0
#     for i in range(len(predicted_order)):
#         if predicted_order[i] == 1:
#             cnt1 += 1
#             ap += (n2 + cnt1) / (i + 1)
#         elif predicted_order[i] == 2:
#             cnt2 += 1
#             ap += (cnt2 / (i + 1))
#     ap /= n
#     return ap


def add_lap(x, eps):
    """[summary]

    Args:
        x ([type]): [description]
        eps ([type]): [description]

    Returns:
        [type]: [description]
    """
    x2 = np.array(x).astype('float64')
    x2 += np.random.laplace(0, 1 / eps, x2.shape)
    x2[x2 < 0] = 0
    return x2


def add_lap_single(x, eps):
    """[summary]

    Args:
        x ([type]): [description]
        eps ([type]): [description]

    Returns:
        [type]: [description]
    """
    x += np.random.laplace(0, 1 / eps)
    return x if x > 0 else 0


def add_guas(x, scale):
    """[summary]

    Args:
        x ([type]): [description]
        scale ([type]): [description]

    Returns:
        [type]: [description]
    """
    x2 = np.array(x).astype('float64')
    x2 += np.random.normal(loc=0, scale=scale, size=x2.shape)
    return x2


def to_string(vectors):
    """[summary]

    Args:
        vectors ([type]): [description]

    Returns:
        [type]: [description]
    """
    string = ""
    for i in range(len(vectors)):
        # string += (str(i + 1) + ':' + str(vectors[i]) + ' ')
        string += (str(vectors[i]) + ' ')
    return string


def evaluation(pred_all, test_relevance_labels, test_query_ids, test_features):
    """[summary]

    Args:
        pred_all ([type]): [description]
        test_relevance_labels ([type]): [description]
        test_query_ids ([type]): [description]
        test_features ([type]): [description]

    Returns:
        [type]: [description]
    """
    _ = test_features
    unique_query_ids = np.unique(test_query_ids)
    ndcg_scores = list()
    full_ndcg_scores = list()
    err_scores = list()
    map_scores = list()
    # print("evaluating...")
    for c_id in unique_query_ids:
        query_indices = np.where(test_query_ids == c_id)[0]
        relevance_labels_onequery = predict_order_for_one_query(
            c_id, query_indices, pred_all, None, test_relevance_labels)
        # print(relevance_labels_onequery)
        # np.random.shuffle(relevance_labels_onequery)
        ndcg_scores.append(ndcg(relevance_labels_onequery, k_pos=10))
        full_ndcg_scores.append(ndcg(relevance_labels_onequery))
        err_scores.append(calc_err(relevance_labels_onequery))
        map_scores.append(cal_map(relevance_labels_onequery))
    avg_ndcg = np.mean(np.array(ndcg_scores))
    avg_full_ndcg = np.mean(np.array(full_ndcg_scores))
    avg_err = np.mean(np.array(err_scores))
    avg_map = np.mean(np.array(map_scores))
    # auc = sklearn.metrics.roc_auc_score(test_relevance_labels, pred_all)
    auc = 0
    print(
        'avg_err',
        avg_err,
        'avg_ndcg',
        avg_ndcg,
        'avg_full_ndcg',
        avg_full_ndcg,
        'avg_map',
        avg_map,
        'auc',
        auc)
    # print('avg_err', avg_err, 'avg_ndcg@10', avg_ndcg, 'avg_full_ndcg', avg_full_ndcg, 'avg_map', avg_map)
    return avg_err, avg_ndcg, avg_full_ndcg, avg_map, auc  # , auc
