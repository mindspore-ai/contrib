"""[summary]
"""
import os
import numpy as np
import tensorflow as tf
from src.utils import evaluation
from src.draw import draw


class GCLSemi:
    """[summary]
    """

    def __init__(self, train_relevance_labels, train_features,
                 test_relevance_labels, test_features, test_query_ids, train_features_u):
        """[summary]

        Args:
            train_relevance_labels ([type]): [description]
            train_features ([type]): [description]
            test_relevance_labels ([type]): [description]
            test_features ([type]): [description]
            test_query_ids ([type]): [description]
            train_features_u ([type]): [description]
        """
        self.y_labeled2 = train_relevance_labels
        self.x_labeled = train_features
        self.x_unlabeled = train_features_u
        self.y_unlabeled = np.zeros([self.x_unlabeled.shape[0], 1])
        self.test_labels = test_relevance_labels
        self.test_features = test_features
        self.test_ids = test_query_ids
        self.n_feature = 0
        self.n_samples = 0
        x = self.x_labeled
        y = self.y_labeled2.reshape(-1, 1)
        x_y = np.concatenate((x, y), axis=1)
        np.random.seed(1)
        np.random.shuffle(x_y)
        self.x_labeled = x_y[:, :-1]
        self.y_labeled2 = x_y[:, -1].reshape(-1,)
        # ------ PARAM -----#
        self.n_point = 40
        self.seed = 37
        self.is_change = False
        self.learning_rate = 0.009
        self.batch_a = 190  # 200 is for GLOBAL+ 500 is for number of party 4
        self.batch_b = 200
        self.lamb = 0.5
        self.beta = 0.
        self.r = 0.2
        self.a = 0.0
        self.af = 0.001
        self.t1 = 0
        self.t2 = 200
        self.n_iter = 300
        # end of param ##

    # def fit(self, from_fed, to_fed, DATA_PATH, FEATURE_NUM=16):
    def fit(self, from_fed, to_fed, _, feature_num=16):
        """[summary]

        Args:
            from_fed ([type]): [description]
            to_fed ([type]): [description]
            DATA_PATH ([type]): [description]
            FEATURE_NUM (int, optional): [description]. Defaults to 16.
        """
        fed_num = to_fed - from_fed
        # initial
        ws1 = np.load(os.path.join("/data/ltrdata", "w1%d.npy" % from_fed))
        ws2 = np.load(os.path.join("/data/ltrdata", "w2%d.npy" % from_fed))
        bs1 = np.load(os.path.join("/data/ltrdata", "b1%d.npy" % from_fed))
        bs2 = np.load(os.path.join("/data/ltrdata", "b2%d.npy" % from_fed))
        for i in range(from_fed + 1, to_fed):
            ws1 += np.load(os.path.join("/data/ltrdata", "w1%d.npy" % i))
            ws2 += np.load(os.path.join("/data/ltrdata", "w2%d.npy" % i))
            bs1 += np.load(os.path.join("/data/ltrdata", "b1%d.npy" % i))
            bs2 += np.load(os.path.join("/data/ltrdata", "b2%d.npy" % i))
        ws1 /= fed_num
        ws2 /= fed_num
        bs1 /= fed_num
        bs2 /= fed_num
        ws = np.load(os.path.join("/data/ltrdata", "semi_ws%d.npy" % from_fed))
        bs = np.load(os.path.join("/data/ltrdata", "semi_bs%d.npy" % from_fed))
        for i in range(from_fed + 1, to_fed):
            ws += np.load(os.path.join("/data/ltrdata", "semi_ws%d.npy" % i))
            bs += np.load(os.path.join("/data/ltrdata", "semi_bs%d.npy" % i))
        ws /= fed_num
        bs /= fed_num
        ws *= 0.1
        bs *= 0.1
        ws += 0.1 * np.random.randn(ws.shape[0], ws.shape[1])
        bs += 0.1 * np.random.randn(bs.shape[0])
        x = tf.placeholder(dtype='float', shape=[None, feature_num], name='x')
        y = tf.placeholder(dtype='float', shape=[None], name='y')
        w = tf.Variable(tf.constant(ws), name='w')
        b = tf.Variable(tf.constant(bs), name='b')
        pred = tf.transpose(tf.add(tf.matmul(x, w), b))
        x_u = tf.placeholder(
            dtype='float', shape=[
                None, feature_num], name='xu')
        pred_u = tf.add(tf.matmul(x_u, w), b)
        pred_us = tf.nn.softmax(tf.add(tf.matmul(tf.add(tf.matmul(x_u, ws1), bs1), ws2), bs2))
        alpha = tf.placeholder("float",)
        pred_pl = tf.placeholder(dtype='float', shape=[None, 1], name='predspl')
        cost = tf.add(self.lamb * tf.reduce_mean(tf.square(w)),
                      tf.add(tf.reduce_mean(tf.square(pred - y) / 2),
                             alpha * tf.reduce_mean(tf.square(pred_pl - pred_u)) / 2))
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.y_unlabeled = sess.run(pred_us, feed_dict={x_u: self.x_unlabeled})
            y_l2 = []
            for each in self.y_unlabeled:
                if each[0] > each[1] and each[0] > each[2]:
                    y_l2.append(0)
                elif each[1] > each[0] and each[1] > each[2]:
                    y_l2.append(1)
                else:
                    y_l2.append(2)
            self.y_unlabeled = np.array(y_l2)
            auc_iters = []
            map_iters = []
            ndcg10_iters = []
            ndcg_iters = []
            err_iters = []
            for it in range(self.n_iter):
                if it > self.t1: a = min((it - self.t1) / (self.t2 - self.t1) * self.af, self.af)
                self.beta /= (1 + 0.5 * it)
                loss_one_fed = []
                x = self.x_labeled
                y = self.y_labeled2.reshape(-1, 1)
                left = it * self.batch_a
                right = left + self.batch_a
                if left >= right or right > len(x):
                    left = 0
                    right = left + self.batch_a
                batch_x = x[left: right]
                batch_y = y[left: right].reshape(-1,)
                x_unlabeled = self.x_unlabeled
                y_unlabeled = self.y_unlabeled
                left = it * self.batch_b
                right = left + self.batch_b
                if left >= right or right > len(x_unlabeled):
                    left = 0
                    right = left + self.batch_b
                batch_x_unlabeled = x_unlabeled[left: right]
                batch_y_unlabeled = y_unlabeled[left: right].reshape(-1, 1)
                if it % (self.n_iter // self.n_point) == 0:
                    pred_for_testo = sess.run(pred, feed_dict={x: self.test_features})[0]
                    print(min(pred_for_testo), max(pred_for_testo), np.mean(pred_for_testo))
                    avg_err, avg_ndcg, avg_full_ndcg, avg_map, avg_auc = \
                        evaluation(pred_for_testo, self.test_labels, self.test_ids, self.test_features)
                    err_iters.append(avg_err)
                    auc_iters.append(avg_auc)
                    map_iters.append(avg_map)
                    ndcg10_iters.append(avg_ndcg)
                    ndcg_iters.append(avg_full_ndcg)
                _, loss = sess.run([opt, cost],
                                   feed_dict={x: batch_x, y: batch_y, x_u: batch_x_unlabeled,
                                              pred_pl: batch_y_unlabeled, alpha: a})
                loss_one_fed.append(loss)
            draw([i for i in range(len(ndcg10_iters))], [ndcg10_iters])
            print("%f, %f, %f, %f;" % (err_iters[-1], ndcg10_iters[-1], ndcg_iters[-1], map_iters[-1]))
            print("nfed_sol4=", ndcg10_iters, ";")
