"""[summary]
"""
import os
import numpy as np
import tensorflow as tf
from src.utils import evaluation
from src.draw import draw


class Semi:
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

    def fit(self, fed_id, data_path, feature_num=16, from_fed=0, to_fed=5):
        """[summary]

        Args:
            fed_id ([type]): [description]
            DATA_PATH ([type]): [description]
            FEATURE_NUM (int, optional): [description]. Defaults to 16.
            from_fed (int, optional): [description]. Defaults to 0.
            to_fed (int, optional): [description]. Defaults to 5.
        """
        _, _, _ = data_path, from_fed, to_fed
        # ------ PARAM -----#
        # local+ af 0.04, 0.4, 0.009 0.3
        # 其他的都一样  60 37 0.01 200 200 0.5 0. 0.15, 0., 0.* 100，200 1000
        n_point = 60
        seed = 37
        learning_rate = 0.01
        batch_a = 200
        batch_b = 200
        lamb = 0.5
        beta = 0.
        r = 0.15
        a = 0.
        af = 0.3
        t1 = 100
        t2 = 200
        n_iter = 1000  # 2000
        # end of param ##
        x = tf.placeholder(dtype='float', shape=[None, feature_num], name='x')
        y = tf.placeholder(dtype='float', shape=[None], name='y')
        w = tf.Variable(tf.random_normal(
            [feature_num, 1], seed=seed), name='w')
        b = tf.Variable(tf.random_normal([1], seed=seed), name='b')
        pred = tf.transpose(tf.add(tf.matmul(x, w), b))
        x_u = tf.placeholder(
            dtype='float', shape=[
                None, feature_num], name='xu')
        pred_u = tf.add(tf.matmul(x_u, w), b)
        alpha = tf.placeholder("float",)
        pred_pl = tf.placeholder(
            dtype='float', shape=[
                None, 1], name='predspl')
        cost = tf.add(lamb * tf.reduce_mean(
            tf.square(w)), tf.add(tf.reduce_mean(tf.square(pred - y) / 2),
                                  alpha * tf.reduce_mean(tf.square(pred_pl - pred_u)) / 2))
        opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        # build initializer
        init = tf.global_variables_initializer()
        # training
        with tf.Session() as sess:
            sess.run(init)
            self.y_unlabeled = sess.run(
                pred_u, feed_dict={
                    x_u: self.x_unlabeled})
            auc_iters = []
            map_iters = []
            ndcg10_iters = []
            ndcg_iters = []
            err_iters = []
            # loss_iters = []
            for it in range(n_iter):
                if it > t1:
                    a = min((it - t1) / (t2 - t1) * af, af)
                beta /= (1 + 2 * it)
                loss_one_fed = []
                x = self.x_labeled
                y = self.y_labeled2.reshape(-1, 1)
                left = it * batch_a
                right = left + batch_a
                if left >= right or right > len(x):
                    left = 0
                    right = left + batch_a
                batch_x = x[left: right]
                batch_y = y[left: right].reshape(-1,)
                x_unlabeled = self.x_unlabeled
                y_unlabeled = self.y_unlabeled
                left = it * batch_b
                right = left + batch_b
                if left >= right or right > len(x_unlabeled):
                    left = 0
                    right = left + batch_b
                batch_x_unlabeled = x_unlabeled[left: right]
                batch_y_unlabeled = y_unlabeled[left: right]
                batch_y_unlabeled = (1 - r) * batch_y_unlabeled + r * sess.run(
                    tf.add(pred_u, beta * 0.5), feed_dict={x_u: batch_x_unlabeled})
                if it % (n_iter // n_point) == 0:
                    pred_for_testo = sess.run(
                        pred, feed_dict={x: self.test_features})[0]
                    avg_err, avg_ndcg, avg_full_ndcg, avg_map, avg_auc = \
                        evaluation(
                            pred_for_testo,
                            self.test_labels,
                            self.test_ids,
                            self.test_features)
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
            # save result
            np.save(
                os.path.join(
                    "/data/ltrdata/",
                    'ws' + str(fed_id)),
                sess.run(w))
            np.save(
                os.path.join(
                    "/data/ltrdata/",
                    'bs' + str(fed_id)),
                sess.run(b))
            print("nfed%d_sol2=" % fed_id, ndcg10_iters, ";")
