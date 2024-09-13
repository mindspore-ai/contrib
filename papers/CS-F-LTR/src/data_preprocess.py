"""[summary]

Returns:
    [type]: [description]
"""
import os
import json
import numpy as np
from sklearn import model_selection, preprocessing
from src.global_variables import FED_NUM, file_path


def load_only(src_path):
    """[summary]

    Args:
        src_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    f = open(src_path)
    ext_label_id_features = []
    real_label_id_features = []
    line = f.readline()
    while line:
        pure_line, comm = line.split('#')  # 去除行尾注释
        pure_line = pure_line.strip()
        comm = comm.strip()[0:3]
        label_id_features = pure_line.split(' ')
        values = []
        for each in label_id_features:
            values.append(json.loads(each))
        if comm in ("non", "rel"):
            real_label_id_features.append(values)
        else:
            ext_label_id_features.append(values)
        line = f.readline()
    f.close()
    return np.array(real_label_id_features), np.array(ext_label_id_features)


def norm(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return preprocessing.normalize(x, norm='l2')


def discrete(y):
    """[summary]

    Args:
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_y = []
    for i in range(len(y)):
        if 0.5 < y[i] <= 10.5:
            new_y.append(2)
        elif 0.5 < y[i] <= 40:
            new_y.append(1)
        else:
            new_y.append(0)
    new_y = np.array(new_y)
    return new_y


def statistics(x):
    """[summary]

    Args:
        x ([type]): [description]
    """
    label2 = 0
    label1 = 0
    label0 = 0
    for each in x:
        if each[0] < 0.5:
            label0 += 1
        elif each[0] < 1.5:
            label1 += 1
        else:
            label2 += 1
    print(len(x), "0:", label0, "1:", label1, "2:", label2)


def split_test_train(label_id_features, test_size=0.3, random_state=0):
    """[summary]

    Args:
        label_id_features ([type]): [description]
        test_size (float, optional): [description]. Defaults to 0.3.
        random_state (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    id_features = label_id_features[:, 1:]
    labels = label_id_features[:, 0]
    id_features_train, id_features_test, labels_train, labels_test = \
        model_selection.train_test_split(id_features, labels, test_size=test_size,
                                         random_state=random_state)
    return id_features_train, id_features_test, labels_train, labels_test

def main():
    for fed_id in range(FED_NUM):
        # real_label_id_features, ext_label_id_features = load_only(
        real_label_id_features, _ = load_only(
            file_path + "fed_std%d.txt" % fed_id)
        np.save(
            os.path.join(
                file_path,
                "real_label_id_features%d.npy" %
                fed_id),
            real_label_id_features)
        np.save(
            os.path.join(
                file_path,
                "ext_label_id_features%d.npy" %
                fed_id),
            real_label_id_features)

if __name__ == '__main__':
    main()
