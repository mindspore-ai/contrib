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
main.py
"""
import argparse

import mindspore as ms
from mindspore import (Tensor, nn)
import numpy as np
from tqdm import tqdm

from dataloader import Dataloader
from evaluate import leave_one_out
from model import LecfNet, LecfLoss, LecfTrainStep, LecfWithLoss


def train(args, data, ms_net):
    '''
    train model
    '''
    avg_loss = 0
    step_number = len(data.true_item) // args.batch_size
    data_iter = data.gen_batch_train_data(args.neg_number, args.batch_size)
    progress = tqdm(enumerate(data_iter), total=(len(data.true_item) // args.batch_size))
    for _, batch in progress:
        batch_user, batch_item, batch_neg_item = batch[:, 0], batch[:, 1], batch[:, 2]
        batch_user = Tensor(batch_user, ms.int32)
        batch_item = Tensor(batch_item, ms.int32)
        batch_neg_item = Tensor(batch_neg_item, ms.int32)

        batch_loss, _ = ms_net(batch_user, batch_item, batch_neg_item, args.depth, args.reg_weight)
        avg_loss += batch_loss
    return ms_net, avg_loss / step_number


def test(args, data, ms_net):
    '''
    test model
    '''
    result = np.zeros([len(list(data.user_test_item.keys())), 4])
    data_iter = data.gen_batch_test_data(args.test_neg_number)
    progress_test = tqdm(enumerate(data_iter), total=data.n_users)
    ms_net.get_propagation_vecs(args.depth)
    for k, e in progress_test:
        purchased_item, batch = e
        batch_user, batch_item = batch[:, 0], batch[:, 1]
        batch_user = Tensor(batch_user, ms.int32)
        batch_item = Tensor(batch_item, ms.int32)

        recommend_list = ms_net.test(batch_user, batch_item, 10)
        recommend_list = recommend_list.asnumpy()

        recommend_list = list(np.array(batch_item)[recommend_list])
        hr5, ndcg5 = leave_one_out(purchased_item, recommend_list, 5)
        hr10, ndcg10 = leave_one_out(purchased_item, recommend_list, 10)
        #
        result[k] = np.array([hr5, ndcg5, hr10, ndcg10])

    avg = np.mean(result, axis=0)
    return avg


def parse_args():
    '''
    get args
    '''
    parser = argparse.ArgumentParser(description="Run CML.")
    parser.add_argument('-d', '--global_dimension', help='Embedding Size', type=int, default=50)
    parser.add_argument('--epochs', help='Max epoch', type=int, default=300)
    parser.add_argument('-n', '--neg_number', help='Negative Samples Count', type=int, default=1)
    parser.add_argument('--test_neg_number', help='Negative Samples Count for testing', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', help='learning_rate', type=float, default=0.002)
    parser.add_argument('--reg_weight', help='wight of l2 Regularization', type=float, default=0.0001)
    parser.add_argument('--dataset', help='path to file', type=str, default='video10')
    parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--outward', type=float, help='outward', default=0.5)

    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()

    # create data
    dataloader = Dataloader()
    filename = './data/' + params.dataset + '/' + params.dataset + '.npz'
    dataloader.get_data(filename)
    adj_mat, mean_adj_mat, edge_adj_mat = dataloader.get_adj_mat(params.outward, params.depth - 1)

    # create model
    item_num = int(dataloader.n_items)
    user_num = int(dataloader.n_users)
    emb_dim = int(params.global_dimension)
    net = LecfNet(item_num, user_num, emb_dim, mean_adj_mat, edge_adj_mat)
    loss_func = LecfLoss()
    opt = nn.Adam(net.trainable_params(), params.learning_rate)
    net_with_criterion = LecfWithLoss(net, loss_func)
    train_net = LecfTrainStep(net_with_criterion, opt)

    # train and test
    for epoch in range(params.epochs):
        train_net, loss = train(params, dataloader, train_net)
        print('epoch:', epoch, 'loss: ', loss)

        results = test(params, dataloader, train_net.network.backbone)
        print('epoch', epoch, 'results:', results)
