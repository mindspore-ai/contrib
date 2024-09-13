# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================
'''evaluate trained net'''

import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.agw import create_agw_net
from src.data.dataset import dataset_creator
from src.utils.config import config
from src.metrics import distance, rank
from src.utils.local_adapter import get_device_id


set_seed(1)


class CustomWithEvalCell(nn.Cell):
    def __init__(self, network):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self._network = network

    def construct(self, data):
        outputs = self._network(data)
        return outputs


def eval_net(net=None):
    '''prepare to eval net'''
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)

    num_train_classes, query_dataset = dataset_creator(
        root=config.data_path, height=config.height, width=config.width,
        dataset=config.target, norm_mean=config.norm_mean,
        norm_std=config.norm_std, batch_size_test=config.batch_size_test,
        workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
        cuhk03_classic_split=config.cuhk03_classic_split,
        mode='query')
    num_train_classes, gallery_dataset = dataset_creator(
        root=config.data_path, height=config.height,
        width=config.width, dataset=config.target,
        norm_mean=config.norm_mean, norm_std=config.norm_std,
        batch_size_test=config.batch_size_test, workers=config.workers,
        cuhk03_labeled=config.cuhk03_labeled,
        cuhk03_classic_split=config.cuhk03_classic_split,
        mode='gallery')

    if net is None:
        net = create_agw_net(num_train_classes)
        param_dict = load_checkpoint(
            config.checkpoint_file_path, filter_prefix='epoch_num')
        params_not_loaded = load_param_into_net(net, param_dict)
        print(params_not_loaded)

    do_eval(net, query_dataset, gallery_dataset)


def do_eval(net, query_dataset, gallery_dataset):
    '''eval the net, called in EvalCallback'''

    net.set_train(False)
    net_eval = CustomWithEvalCell(net)

    def feature_extraction(eval_dataset):
        f_, pids_, camids_ = [], [], []
        for data in eval_dataset.create_dict_iterator():
            imgs, pids, camids = data['img'], data['pid'], data['camid']
            features = net_eval(imgs)
            f_.append(features)
            pids_.extend(pids.asnumpy())
            camids_.extend(camids.asnumpy())
        concat = ops.Concat(axis=0)
        f_ = concat(f_)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return f_, pids_, camids_

    print('Extracting features from query set ...')
    qf, q_pids, q_camids = feature_extraction(query_dataset)
    print('Done, obtained {}-by-{} matrix'.format(qf.shape[0], qf.shape[1]))

    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = feature_extraction(gallery_dataset)
    print('Done, obtained {}-by-{} matrix'.format(gf.shape[0], gf.shape[1]))

    if config.normalize_feature:
        l2_normalize = ops.L2Normalize(axis=1)
        qf = l2_normalize(qf)
        gf = l2_normalize(gf)

    print('Computing distance matrix with metric={} ...'.format(config.dist_metric))
    distmat = distance.compute_distance_matrix(qf, gf, config.dist_metric)
    distmat = distmat.asnumpy()

    if not config.use_metric_cuhk03:
        print('Computing CMC mAP mINP ...')
        cmc, mean_ap, mean_inp = rank.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=config.use_metric_cuhk03
        )
    else:
        print('Computing CMC and mAP ...')
        cmc, mean_ap = rank.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=config.use_metric_cuhk03
        )

    print('** Results **')
    print('ckpt={}'.format(config.checkpoint_file_path))
    print('mAP: {:.2%}'.format(mean_ap))
    print('mINP: {:.2%}'.format(mean_inp))
    print('CMC curve')
    ranks = [1, 5, 10, 20]
    i = 0
    for r in ranks:
        print('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
        i += 1


if __name__ == '__main__':
    eval_net()
