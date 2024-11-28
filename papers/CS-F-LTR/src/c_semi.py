"""[summary]
"""
import os
import numpy as np
import tensorflow as tf
from src.utils import evaluation
from src.draw import draw


class SemiClassifier:
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
        self.y_labeled3 = []
        for each in self.y_labeled2:
            if each < 0.5:
                self.y_labeled3.append([1, 0, 0])
            elif each < 1.5:
                self.y_labeled3.append([0, 1, 0])
            else:
                self.y_labeled3.append([0, 0, 1])
        self.y_labeled3 = np.array(self.y_labeled3).reshape(-1, 3)
        # ------ PARAM -----#
        # local+ af 0.04, 0.4, 0.009 0.3
        # 其他的都一样  60 37 0.01 200 200 0.5 0. 0.15, 0., 0.* 100，200 1000
        self.n_point = 30
        self.seed = 37
        self.learning_rate = 0.01
        self.batch_a = 300
        self.batch_b = 200
        self.lamb = 0.
        self.beta = 0.
        self.r = 0.15
        self.a = 0.
        self.af = 0.01
        self.t1 = 200
        self.t2 = 300
        self.n_iter = 500  # 2000
        # end of param ##

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
        x = tf.placeholder(dtype='float', shape=[None, feature_num], name='x')
        y = tf.placeholder(dtype='float', shape=[None, 3], name='y')
        w1 = tf.Variable(tf.random_normal([feature_num, 20], seed=self.seed), name='w1')
        b1 = tf.Variable(tf.random_normal([20], seed=self.seed), name='b1')
        h1 = tf.add(tf.matmul(x, w1), b1)
        w2 = tf.Variable(tf.random_normal([20, 3], seed=self.seed), name='w2')
        b2 = tf.Variable(tf.random_normal([3], seed=self.seed), name='b2')
        pred = tf.nn.softmax(tf.add(tf.matmul(h1, w2), b2))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        x_u = tf.placeholder(dtype='float', shape=[None, feature_num], name='xu')
        pred_u = tf.nn.softmax(tf.add(tf.matmul(tf.add(tf.matmul(x_u, w1), b1), w2), b2))
        alpha = tf.placeholder("float",)
        pred_pl = tf.placeholder(dtype='float', shape=[None, 3], name='predspl')
        cost = tf.add(
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)),
            alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred_pl, logits=pred_u))
        )
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        # build initializer
        init = tf.global_variables_initializer()
        # training
        with tf.Session() as sess:
            sess.run(init)
            self.y_unlabeled = sess.run(pred_u, feed_dict={x_u: self.x_unlabeled})
            auc_iters = []
            map_iters = []
            ndcg10_iters = []
            ndcg_iters = []
            err_iters = []
            for it in range(self.n_iter):
                if it > self.t1:
                    a = min((it - self.t1) / (self.t2 - self.t1) * self.af, self.af)
                self.beta /= (1 + 2 * it)
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
                batch_y2 = []
                for each in batch_y:
                    if each == 0:
                        batch_y2.append([1, 0, 0])
                    elif each == 1:
                        batch_y2.append([0, 1, 0])
                    else:
                        batch_y2.append([0, 0, 1])
                batch_y = np.array(batch_y2)
                x_unlabeled = self.x_unlabeled
                y_unlabeled = self.y_unlabeled
                left = it * self.batch_b
                right = left + self.batch_b
                if left >= right or right > len(x_unlabeled):
                    left = 0
                    right = left + self.batch_b
                batch_x_unlabeled = x_unlabeled[left: right]
                batch_y_unlabeled = y_unlabeled[left: right]
                batch_y_unlabeled = (1 - self.r) * batch_y_unlabeled + self.r * sess.run(
                    tf.add(pred_u, self.beta * 0.5), feed_dict={x_u: batch_x_unlabeled})
                if it % (self.n_iter // self.n_point) == 0:
                    pred_for_testo = sess.run(
                        pred, feed_dict={x: self.test_features})
                    pred_for_testo2 = []
                    for each in pred_for_testo:
                        if each[0] > each[1] and each[0] > each[2]:
                            pred_for_testo2.append(0)
                        elif each[1] > each[0] and each[1] > each[2]:
                            pred_for_testo2.append(1)
                        else:
                            pred_for_testo2.append(2)
                    pred_for_testo = np.array(pred_for_testo2)
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
            self.save_result(sess, fed_id, w1, w2, b1, b2)
            print("nfed%d_sol2=" % fed_id, ndcg10_iters, ";")

    def save_result(self, sess, fed_id, w1, w2, b1, b2):
        """[summary]

        Args:
            sess ([type]): [description]
            fed_id ([type]): [description]
            w1 ([type]): [description]
            w2 ([type]): [description]
            b1 ([type]): [description]
            b2 ([type]): [description]
        """
        # save result
        np.save(
            os.path.join(
                "/data/ltrdata/",
                'w1' + str(fed_id)),
            sess.run(w1))
        np.save(
            os.path.join(
                "/data/ltrdata/",
                'w2' + str(fed_id)),
            sess.run(w2))
        np.save(
            os.path.join(
                "/data/ltrdata/",
                'b1' + str(fed_id)),
            sess.run(b1))
        np.save(
            os.path.join(
                "/data/ltrdata/",
                'b2' + str(fed_id)),
            sess.run(b2))
