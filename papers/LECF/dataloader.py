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
dataloader.py
"""
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


class Data():
    '''
    dataloader for LECF
    '''
    def __init__(self):
        self.user_number = 0
        self.item_number = 0

        self.user_item = defaultdict(set)
        self.item_user = defaultdict(set)

        self.true_user = []
        self.true_item = []
        self.true_number = 0

        self.user_vali_item = dict()
        self.user_test_item = dict()

    def get_data(self, npzpath):
        '''
        generate data
        '''
        data = np.load(npzpath, allow_pickle=True)
        train_data = data['train_data']
        test_data = data['test_data'].tolist()
        vali_data = data['vali_data'].tolist()

        p = npzpath.split('/')
        self.path = p[0] + '/' + p[1] + '/' + p[2]

        self.n_users, self.n_items = train_data.max(axis=0) + 1
        self.r = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        for u, i in train_data:
            self.user_item[u].add(i)
            self.item_user[i].add(u)

            self.true_user.append(u)
            self.true_item.append(i)
            self.r[u, i] = 1.

        self.true_number = len(self.true_user)

        for u in test_data.keys():
            self.user_test_item[u] = test_data[u][1]
            self.user_test_item[u].append(test_data[u][0])

        for u in vali_data.keys():
            self.user_vali_item[u] = vali_data[u][1]
            self.user_vali_item[u].extend([vali_data[u][0]])

    def shuffle(self):
        '''
        shuffle data
        '''
        index = np.array(range(len(self.true_item)))
        np.random.shuffle(index)
        self.true_user = np.array(self.true_user)[index]
        self.true_item = np.array(self.true_item)[index]

    def get_adj_mat(self, c, l):
        '''
        load matrix
        :param c: outward parameter
        :param l: depth of walk
        :return:
        '''

        adj_mat, mean_adj_mat, edge_adj_mat = self.create_adj_mat(c, l)
        sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
        sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        sp.save_npz(self.path + '/s_edge_adj_mat-{}-{}.npz'.format(c, l), edge_adj_mat)

        return adj_mat.todense(), mean_adj_mat.todense(), edge_adj_mat.todense()

    def create_adj_mat(self, c, l):
        '''
        create matrix
        :param c: outward parameter
        :param l: depth of walk
        :return:
        '''
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        r = self.r.tolil()

        adj_mat[:self.n_users, self.n_users:] = r
        adj_mat[self.n_users:, :self.n_users] = r.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def edge_mat(adj, c, l):
            if l == 0:
                return adj
            mat = c * adj + (1 - c) * sp.eye(adj.shape[0])
            for i in range(l):
                if i == 0:
                    edge_mat = mat.dot(mat)
                else:
                    edge_mat = edge_mat.dot(mat)
            return edge_mat.tocoo()

        mean_adj_mat = normalized_adj_single(adj_mat)
        edge_adj_mat = edge_mat(mean_adj_mat, c, l)

        return adj_mat.tocsr(), mean_adj_mat.tocsr(), edge_adj_mat.tocsr()


class Dataloader(Data):
    '''
    Dataloader
    '''
    def gen_batch_train_data(self, neg_number, batch_size):
        '''
        train data
        '''
        self.shuffle()

        batch = np.zeros((batch_size, 3), dtype=np.uint32)

        idx = 0
        for i in range(len(self.true_user)):
            for _ in range(neg_number):
                neg_item = random.randint(0, self.n_items - 1)
                while neg_item in self.user_item[self.true_user[i]]:
                    neg_item = random.randint(0, self.n_items - 1)

                batch[idx, :] = [self.true_user[i], self.true_item[i], neg_item]
                idx += 1

                if idx == batch_size:
                    yield batch
                    idx = 0

        if idx > 0:
            yield batch

    def gen_batch_test_data(self, test_neg_number):
        '''
        test data
        '''
        size = test_neg_number + 1
        batch = np.zeros((size, 2), dtype=np.uint32)

        idx = 0
        for user, items in self.user_test_item.items():
            for item in items:
                batch[idx, :] = [user, item]
                idx += 1

            yield items[-1], batch
            idx = 0

    def gen_batch_vali_data(self, test_neg_number):
        '''
        vali datae
        '''
        size = test_neg_number + 1
        batch = np.zeros((size, 2), dtype=np.uint32)

        idx = 0
        for user, items in self.user_vali_item.items():
            for item in items:
                batch[idx, :] = [user, item]
                idx += 1

            yield items[-1], batch
            idx = 0
