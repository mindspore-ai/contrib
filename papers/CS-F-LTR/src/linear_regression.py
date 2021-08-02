"""[summary]

Returns:
    [type]: [description]
"""
#!/usr/bin/env python
# coding: utf-8

from mindspore import context
from mindspore import dataset as ds
from mindspore import nn
from mindspore import Tensor
from mindspore import Model
from mindspore import ops
from mindspore.train.callback import Callback

from src.utils import evaluation
# from src.draw import draw

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')


class LossFn(nn.Cell):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, lamb, weight):
        """[summary]

        Args:
            lamb ([type]): [description]
            weight ([type]): [description]
        """
        super().__init__()
        self.lamb = lamb
        self.weight = weight[0]
        self.square = ops.Square()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, x, y):
        """[summary]

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.lamb * \
            self.reduce_mean(self.square(self.weight)) + \
            self.reduce_mean(self.square(x - y) / 2)


class LinearNet(nn.Cell):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self):
        """[summary]
        """
        super().__init__()
        self.fc = nn.Dense(8, 1)

    def construct(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = self.fc(x)
        return x


class EvalCallBack(Callback):
    """[summary]

    Args:
        Callback ([type]): [description]
    """
    def __init__(self, net, test_features, test_labels,
                 test_ids, eval_per_epoch, metric_data):
        """[summary]

        Args:
            net ([type]): [description]
            test_features ([type]): [description]
            test_labels ([type]): [description]
            test_ids ([type]): [description]
            eval_per_epoch ([type]): [description]
            metric_data ([type]): [description]
        """
        self.net = net
        self.test_features = Tensor(test_features)
        self.test_labels = test_labels
        self.test_ids = test_ids
        self.eval_per_epoch = eval_per_epoch
        self.metric_data = metric_data

    def epoch_end(self, run_context):
        """[summary]

        Args:
            run_context ([type]): [description]
        """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        print('epoch %d' % cur_epoch)
        if cur_epoch % self.eval_per_epoch == 0:
            pred_for_testo = self.net(self.test_features).asnumpy()
            avg_err, avg_ndcg, avg_full_ndcg, avg_map, avg_auc = evaluation(
                pred_for_testo, self.test_labels, self.test_ids, self.test_features)
            self.metric_data['err_iters'].append(avg_err)
            self.metric_data['auc_iters'].append(avg_auc)
            self.metric_data['map_iters'].append(avg_map)
            self.metric_data['ndcg10_iters'].append(avg_ndcg)
            self.metric_data['ndcg_iters'].append(avg_full_ndcg)


class LinearRegression():
    """[summary]
    """

    def __init__(self, train_relevance_labels, train_features,
                 test_relevance_labels, test_features, test_query_ids):
        """[summary]

        Args:
            train_relevance_labels ([type]): [description]
            train_features ([type]): [description]
            test_relevance_labels ([type]): [description]
            test_features ([type]): [description]
            test_query_ids ([type]): [description]
        """
        # input is numpy array
        self.train_labels = train_relevance_labels.reshape(-1, 1)
        self.train_features = train_features
        self.test_labels = test_relevance_labels.reshape(-1, 1)
        self.test_features = test_features
        self.test_ids = test_query_ids
        self.n_feature = 0
        self.n_samples = 0

    def fit(self, fed_id, fn, data_path=""):
        """[summary]

        Args:
            fed_id ([type]): [description]
            fn (function): [description]
            DATA_PATH (str, optional): [description]. Defaults to "".
        """
        _, _, _ = fed_id, fn, data_path
        # FEATURE_NUM = fn
        # shape = [FEATURE_NUM, 1]
        learning_rate = 0.02
        batch_a = 500
        lamb = 0.5
        # n_iter = 2000
        # seed = 30
        # n_point = 40
        epoch = 50
        # -- END OF PARAM --#
        # build model
        ds_train = ds.GeneratorDataset(list(
            zip(self.train_features, self.train_labels)), column_names=['data', 'label'])
        ds_train = ds_train.batch(batch_a)
        net = LinearNet()
        net_loss = LossFn(lamb, net.trainable_params())
        opt = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
        # training
        auc_iters = []
        map_iters = []
        ndcg10_iters = []
        ndcg_iters = []
        err_iters = []
        metric_data = {
            'auc_iters': auc_iters,
            'map_iters': map_iters,
            'ndcg10_iters': ndcg10_iters,
            'ndcg_iters': ndcg_iters,
            'err_iters': err_iters,
        }
        model = Model(net, net_loss, opt)
        eval_callback = EvalCallBack(
            net,
            self.test_features,
            self.test_labels,
            self.test_ids,
            1,
            metric_data)
        model.train(
            epoch,
            ds_train,
            callbacks=[eval_callback],
            dataset_sink_mode=False)

        #draw([i for i in range(len(ndcg10_iters))], [ndcg10_iters])
        #print("nfed%d_sol1="% fed_id, ndcg10_iters, ";")
        #print(f"{err_iters[-1]:f} {ndcg10_iters[-1]:f} {ndcg_iters[-1]:f} {map_iters[-1]:f} {auc_iters[-1]:f}")
