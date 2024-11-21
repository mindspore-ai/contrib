import mindspore
import mindspore.numpy as mnp
import numpy as np
from mindspore import Tensor
import scipy.sparse as sp
import mindspore.ops as op

def scipy_sparse_mat_to_mindspore_sparse_tensor(sparse_mx):
    # 转为CSR矩阵
    sparse_mx = sparse_mx.tocoo().astype('float32')
    # 索引
    shape = sparse_mx.shape
    indptr = mnp.zeros(shape=shape[0]+1).astype('int64')

    for i in range(len(sparse_mx.row)):
        row = int(sparse_mx.row[i])
        indptr[row+1] = i+1
    for i in range(1,len(indptr)):
        if indptr[i] < indptr[i-1]:
            indptr[i] = indptr[i-1]
    # 转为mindspore tensor
    indptr = Tensor(indptr,dtype=mindspore.int32)
    indices = Tensor(sparse_mx.col,dtype=mindspore.int32)
    values = Tensor(sparse_mx.data, mindspore.float32)
    

    return mindspore.CSRTensor(indptr, indices, values, shape)

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = 0
            for loc in range(min(topk, len(label))):
                idcg = idcg + 1 / np.log2(loc + 2)
            dcg = 0
            for item in label:
                if item in prediction:
                    hit += 1
                    loc = prediction.index(item)
                    dcg = dcg + 1 / np.log2(loc + 2)
            all_recall = all_recall + hit / len(label)
            all_ndcg = all_ndcg + dcg / idcg
            user_num += 1
    return all_recall / user_num, all_ndcg / user_num