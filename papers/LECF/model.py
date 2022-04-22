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
model.py
"""
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, nn


class LecfNet(nn.Cell):
    '''
    LECF
    '''
    def __init__(self, item_num, user_num, emb_dim, mean_adj_mat, edge_adj_mat):
        super(LecfNet, self).__init__()

        self.item_num, self.user_num = item_num, user_num
        self.item_matrix = nn.Embedding(item_num, emb_dim, True)
        self.user_matrix = nn.Embedding(user_num, emb_dim, True)

        self.edge_adj_mat = edge_adj_mat

        self.sparse_dense_matmul = nn.SparseTensorDenseMatmul()
        self.n_fold = 10
        self.n_users, self.n_items = user_num, item_num

        self.mean_adj_mat = Tensor(mean_adj_mat, ms.float32)
        self.edge_adj_mat = Tensor(edge_adj_mat, ms.float32)

        self.all_items = Tensor(list(range(self.item_num)))
        self.all_users = Tensor(list(range(self.user_num)))

        self.opk = ops.TopK(sorted=True)

    def propagation(self, depth):
        '''
        create graph
        '''
        all_user_emb = self.user_matrix(self.all_users)
        all_item_emb = self.item_matrix(self.all_items)
        ego_embeddings = ops.Concat(axis=0)((all_user_emb, all_item_emb))

        ego_embeddings = ops.matmul(self.mean_adj_mat, ego_embeddings) + ego_embeddings
        ego_embeddings = ego_embeddings * 0.5

        if depth > 1:
            ego_embeddings = ops.matmul(self.edge_adj_mat, ego_embeddings)

        u_g_embeddings, i_g_embeddings = ego_embeddings[:self.n_users], ego_embeddings[self.n_users:]
        return u_g_embeddings, i_g_embeddings

    def construct(self, user_ids, item_ids, neg_item_ids, depth):
        '''
        forward
        '''
        user_emb = self.user_matrix(user_ids)
        item_emb = self.item_matrix(item_ids)
        neg_emb = self.item_matrix(neg_item_ids)

        u_g_embeddings, i_g_embeddings = self.propagation(depth=depth)

        axis = 0
        last_user_embedding = ops.Gather()(u_g_embeddings, user_ids, axis)
        last_item_embedding = ops.Gather()(i_g_embeddings, item_ids, axis)
        last_neg_embedding = ops.Gather()(i_g_embeddings, neg_item_ids, axis)

        y = ops.ReduceSum()(ops.Mul()(user_emb + item_emb, last_user_embedding + last_item_embedding), 1)
        neg_y = ops.ReduceSum()(ops.Mul()(user_emb + neg_emb, last_user_embedding + last_neg_embedding), 1)

        return y, neg_y, user_emb, item_emb, neg_emb

    def get_propagation_vecs(self, depth):
        '''
        propagation
        '''
        self.u_g_embeddings, self.i_g_embeddings = self.propagation(depth=depth)

    def test(self, user_ids, item_ids, topk):
        '''
        test results
        '''
        user_emb = self.user_matrix(user_ids)
        item_emb = self.item_matrix(item_ids)

        axis = 0
        last_user_embedding = ops.Gather()(self.u_g_embeddings, user_ids, axis)
        last_item_embedding = ops.Gather()(self.i_g_embeddings, item_ids, axis)

        y = ops.ReduceSum()(ops.Mul()(user_emb + item_emb, last_user_embedding + last_item_embedding), 1)
        _, indices = self.opk(y, topk)
        return indices


class LecfLoss(nn.Cell):
    '''
    loss function
    '''
    def construct(self, y, neg_y, user_emb, item_emb, neg_emb, reg_weight):
        '''
        forward
        '''
        mf_loss = - ops.ReduceSum()(ops.Log()(nn.Sigmoid()(y - neg_y))) / y.shape[0]
        reg_loss = ops.ReduceSum()(ops.Mul()(user_emb, user_emb)) \
                   + ops.ReduceSum()(ops.Mul()(item_emb, item_emb)) \
                   + ops.ReduceSum()(ops.Mul()(neg_emb, neg_emb))
        reg_loss = reg_weight * reg_loss / y.shape[0]
        return mf_loss + reg_loss


class LecfWithLoss(nn.Cell):
    '''
    Lecf with loss
    '''
    def __init__(self, backbone, loss_fn):
        super(LecfWithLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, user_ids, item_ids, neg_item_ids, depth, reg_weight):
        '''
        forward
        '''
        y, neg_y, user_emb, item_emb, neg_emb = self.backbone(user_ids, item_ids, neg_item_ids, depth)
        return self.loss_fn(y, neg_y, user_emb, item_emb, neg_emb, reg_weight)

    def backbone_network(self):
        return self.backbone


class LecfTrainStep(nn.TrainOneStepCell):
    '''
    LECF Train model
    '''
    def __init__(self, network, optimizer):
        super(LecfTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, user_ids, item_ids, neg_item_ids, depth, reg_weight):
        '''
        forward
        '''
        weights = self.weights
        loss = self.network(user_ids, item_ids, neg_item_ids, depth, reg_weight)
        grads = self.grad(self.network, weights)(user_ids, item_ids, neg_item_ids, depth, reg_weight)
        return loss, self.optimizer(grads)
