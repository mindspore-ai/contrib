"""[summary]
"""
import pickle
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils import evaluation


class DecisionTree:
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
        x = self.x_labeled
        y = self.y_labeled2.reshape(-1, 1)
        x_y = np.concatenate((x, y), axis=1)
        np.random.seed(1)
        np.random.shuffle(x_y)
        self.x_labeled = x_y[:, :-1]
        self.y_labeled2 = x_y[:, -1].reshape(-1,)

    def fit(self, fed_id, file_path):
        """[summary]

        Args:
            fed_id ([type]): [description]
            file_path ([type]): [description]
        """
        clf = DecisionTreeClassifier()
        clf.fit(self.x_labeled, self.y_labeled2)
        result = clf.predict(self.test_features)
        # use label only
        # avg_err, avg_ndcg, avg_full_ndcg, avg_map, avg_auc = \
        _, _, _, _, _ = \
            evaluation(
                result,
                self.test_labels,
                self.test_ids,
                self.test_features)

        pickle.dump(
            clf,
            open(
                os.path.join(
                    file_path,
                    "decision_tree%d" %
                    fed_id),
                "wb"))
