import time
import pickle
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as op
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer
from parser import args
from utils import scipy_sparse_mat_to_mindspore_sparse_tensor, metrics as mindspore_metrics 


mindspore.set_context(mode=mindspore.GRAPH_MODE)  

k = args.k

# 加载数据
with open('data/' + args.data + '/train_mat.pkl', 'rb') as f:
    train = pickle.load(f)
with open('data/' + args.data + '/test_mat.pkl', 'rb') as f:
    test = pickle.load(f)

test_labels = [[] for _ in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Data loaded and processed.')

start_time = time.time()

# 加载矩阵
n_u, n_i = train.shape

item_rep = mindspore.ops.eye(n_i, n_i, mindspore.float32)
user_rep = mindspore.Tensor(np.zeros((n_u, n_i)), mindspore.float32)

# 对稀疏矩阵进行转换
adj = scipy_sparse_mat_to_mindspore_sparse_tensor(train)
adj_t = adj.to_dense().t().to_csr()
# 迭代表示在图上的传播
for i in range(k):
    print("Running layer", i)
    
    user_rep_temp = adj.mm(item_rep) + user_rep
    item_rep_temp = adj_t.mm(user_rep) + item_rep
    user_rep = user_rep_temp
    item_rep = item_rep_temp

# 评估
pred = user_rep.asnumpy()

train_csr = (train!= 0).astype(np.float32)

batch_user = 256
test_uids = np.array([i for i in range(test.shape[0])])
batch_no = int(np.ceil(len(test_uids) / batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch * batch_user
    end = min((batch + 1) * batch_user, len(test_uids))

    preds = pred[start:end]
    mask = train_csr[start:end].toarray()
    preds = preds * (1 - mask)
    predictions = (-preds).argsort()

    # top@20
    recall_20, ndcg_20 = mindspore_metrics(test_uids[start:end], predictions, 20, test_labels)
    # top@40
    recall_40, ndcg_40 = mindspore_metrics(test_uids[start:end], predictions, 40, test_labels)

    all_recall_20 += recall_20
    all_ndcg_20 += ndcg_20
    all_recall_40 += recall_40
    all_ndcg_40 += ndcg_40
    print('batch', batch, 'recall@20', recall_20, 'ndcg@20', ndcg_20, 'recall@40', recall_40, 'ndcg@40', ndcg_40)
print('-------------------------------------------')
print('recall@20', all_recall_20 / batch_no, 'ndcg@20', all_ndcg_20 / batch_no, 'recall@40', all_recall_40 / batch_no,
      'ndcg@40', all_ndcg_40 / batch_no)

end_time = time.time()
print("Total running time (seconds):", end_time - start_time)