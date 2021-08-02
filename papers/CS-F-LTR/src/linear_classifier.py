"""[summary]
"""
import tensorflow as tf
import numpy as np
from src.utils import evaluation
from src.draw import draw


class Classifier:
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
        self.y_labeled2 = train_relevance_labels
        self.x_labeled = train_features
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
                self.y_labeled3.append([0, 0, 1])
            elif each < 1.5:
                self.y_labeled3.append([0, 1, 0])
            else:
                self.y_labeled3.append([1, 0, 0])
        self.y_labeled3 = np.array(self.y_labeled3).reshape(-1, 3)

    def fit(self, fed_id, fn=51):
        """[summary]

        Args:
            fed_id ([type]): [description]
            fn (int, optional): [description]. Defaults to 51.
        """
        # ------ PARAM -----#
        # local: 0.01  100
        learning_rate = 0.01
        batch_a = 600
        n_iter = 1000
        seed = 20
        n_point = 30
        # -- END OF PARAM --#
        # build model
        x = tf.placeholder(dtype='float', shape=[None, fn], name='x')
        y = tf.placeholder(dtype='float', shape=[None, 3], name='y')
        w1 = tf.Variable(tf.random_normal([fn, 20], seed=seed), name='w1')
        b1 = tf.Variable(tf.random_normal([20], seed=seed), name='b1')
        h1 = tf.add(tf.matmul(x, w1), b1)
        w2 = tf.Variable(tf.random_normal([20, 3], seed=seed), name='w2')
        b2 = tf.Variable(tf.random_normal([3], seed=seed), name='b2')
        pred = tf.nn.softmax(tf.add(tf.matmul(h1, w2), b2))
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y, logits=pred))
        opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        # training
        with tf.Session() as sess:
            sess.run(init)
            auc_iters = []
            map_iters = []
            ndcg10_iters = []
            ndcg_iters = []
            err_iters = []
            # loss_iters = []
            for it in range(n_iter):
                # loss_one_fed = []
                x = self.x_labeled
                y = self.y_labeled3
                left = it * batch_a
                right = left + batch_a
                if left >= right or right > len(x):
                    left = 0
                    right = left + batch_a
                batch_x = x[left: right]
                batch_y = y[left: right]
                if it % (n_iter // n_point) == 0:
                    pred_for_testo = sess.run(
                        pred, feed_dict={x: self.test_features})
                    pred_test_compact = []
                    for each in pred_for_testo:
                        if each[0] > each[1] and each[0] > each[2]:
                            pred_test_compact.append(2)
                        elif each[1] > each[0] and each[1] > each[2]:
                            pred_test_compact.append(1)
                        else:
                            pred_test_compact.append(0)
                    pred_test_compact = np.array(pred_test_compact)
                    # print(min(pred_for_testo), max(pred_for_testo))
                    avg_err, avg_ndcg, avg_full_ndcg, avg_map, avg_auc = \
                        evaluation(
                            pred_test_compact,
                            self.test_labels,
                            self.test_ids,
                            self.test_features)
                    err_iters.append(avg_err)
                    auc_iters.append(avg_auc)
                    map_iters.append(avg_map)
                    ndcg10_iters.append(avg_ndcg)
                    ndcg_iters.append(avg_full_ndcg)
                # _, loss = sess.run([opt, cost], feed_dict={
                _, _ = sess.run([opt, cost], feed_dict={
                    x: batch_x, y: batch_y})
        draw([i for i in range(len(ndcg10_iters))], [ndcg10_iters])
        print("nfed%d_sol1=" % fed_id, ndcg10_iters, ";")
        print(
            f"{err_iters[-1]:f} {ndcg10_iters[-1]:f} {ndcg_iters[-1]:f} {map_iters[-1]:f} {auc_iters[-1]:f}")
