"""[summary]
"""
#! /usr/bin/env python3
# coding=UTF-8
import sys
import pickle
import numpy as np
from sklearn import preprocessing
from src.linear_regression import LinearRegression
from src.decision_tree import DecisionTree
from src.c_semi import SemiClassifier
from src.global_variables import FED_NUM, file_path
from src.data_preprocess import split_test_train, statistics

try:
    TRAIN_MODE = int(sys.argv[1])
except ValueError:
    TRAIN_MODE = 1

all_real_label_id_features = []
all_ext_label_id_features = []
wrap_real_label_id_features = []
wrap_ext_label_id_features = []

all_train_y = []
all_train_x = []
all_train_id = []

all_train_x_u = []
all_train_y_u = []
all_train_id_u = []

all_test_y = []
all_test_x = []
all_test_id = []


for i in range(FED_NUM):
    real_label_id_features, ext_label_id_features = pickle.load(
        open("/data/icde_data/fed_with_sh_(5_30_200)ext%d.pkl" % i, "rb"))
    np.random.seed(0)
    np.random.shuffle(real_label_id_features)

    statistics(real_label_id_features)
    statistics(ext_label_id_features)

    real_id_x_train, real_id_x_test, real_y_train, real_y_test = split_test_train(
        real_label_id_features, test_size=0.3, random_state=0)

    all_train_x.append(preprocessing.scale(real_id_x_train[:, 1:]))
    all_train_id.append(real_id_x_train[:, 0])
    all_train_y.append(real_y_train)

    all_train_x_u.append(preprocessing.scale(ext_label_id_features[:, 2:]))
    all_train_y_u.append(ext_label_id_features[:, 0])
    all_train_id_u.append(ext_label_id_features[:, 1])

    all_test_y.append(real_y_test)
    all_test_id.append(real_id_x_test[:, 0])
    all_test_x.append(preprocessing.scale(real_id_x_test[:, 1:]))
print("test:", len(all_test_x[0]))
from_fed = 0
to_fed = 4  # FED_NUM

if TRAIN_MODE == 5:
    # Decision Tree
    for i in range(from_fed, to_fed):
        model = DecisionTree(
            all_train_y[i],
            all_train_x[i],
            all_test_y[i],
            all_test_x[i],
            all_test_id[i])
        model.fit(i, "/data/icde_data")


if TRAIN_MODE in [1, 3, 6]:
    # LOCAL
    for i in range(from_fed, to_fed):
        if TRAIN_MODE == 1:
            model = LinearRegression(
                all_train_y[i],
                all_train_x[i],
                all_test_y[i],
                all_test_x[i],
                all_test_id[i])
            model.fit(i, 8)
        elif TRAIN_MODE == 6:
            all_train_y[i] = np.concatenate(
                (all_train_y[i], all_train_y_u[i]), axis=0)
            all_train_x[i] = np.concatenate(
                (all_train_x[i], all_train_x_u[i]), axis=0)
            model = LinearRegression(
                all_train_y[i],
                all_train_x[i],
                all_test_y[i],
                all_test_x[i],
                all_test_id[i])
            model.fit(i, 8)
        else:
            # LOCAL+
            model = SemiClassifier(
                all_train_y[i],
                all_train_x[i],
                all_test_y[i],
                all_test_x[i],
                all_test_id[i],
                all_train_x_u[i])
            model.fit(i, file_path)

elif TRAIN_MODE in [2, 4]:
    train_f = []
    train_l = []
    ext_f = []
    ext_l = []
    test_f = []
    test_l = []
    test_id = []
    for i in range(from_fed, to_fed):
        train_f.extend(all_train_x[i])
        train_l.extend(all_train_y[i])
        ext_f.extend(all_train_x_u[i])
        ext_l.extend(all_train_y_u[i])
        test_f.extend(all_test_x[i])
        test_l.extend(all_test_y[i])
        test_id.extend(all_test_id[i])
    train_f = np.array(train_f)
    train_l = np.array(train_l).reshape(-1, 1)
    train_f_l = np.concatenate((train_f, train_l), axis=1)
    np.random.shuffle(train_f_l)
    train_f = train_f_l[:, :-1]
    train_l = train_f_l[:, -1]
    ext_f = np.array(ext_f)
    np.random.shuffle(ext_f)

    test_f = np.array(test_f)
    test_l = np.array(test_l).reshape(-1, 1)
    test_id = np.array(test_id).reshape(-1, 1)
    test_id_f_l = np.concatenate((test_id, test_f, test_l), axis=1)
    np.random.shuffle(test_id_f_l)
    total_num = test_id_f_l.shape[0]
    per_num = total_num // (to_fed - from_fed)
    test_id = test_id_f_l[:per_num, 0]
    test_f = test_id_f_l[:per_num, 1:-1]
    test_l = test_id_f_l[:per_num, -1]

    if TRAIN_MODE == 2:
        # GLOBAL
        model = LinearRegression(
            train_l,
            train_f,
            np.array(test_l),
            np.array(test_f),
            np.array(test_id))
        model.fit(FED_NUM, 8, file_path)
        # model = Classifier(train_l, train_f, np.array(test_l), np.array(test_f), np.array(test_id))
        # model.fit(FED_NUM, 16)
        # model = DecisionTree(train_l, train_f, np.array(test_l), np.array(test_f), np.array(test_id))
        # model.fit(FED_NUM, "/data/icde_data")

    else:
        # GLOBAL+
        # model = GCSemi(train_l, train_f, np.array(test_l), np.array(test_f), np.array(test_id), ext_f)
        # model.fit(FED_NUM, file_path, 16)
        # model = GCLSemi(train_l, train_f, np.array(test_l), np.array(test_f), np.array(test_id), ext_f)
        # model.fit(0, FED_NUM, file_path, 16)
        # model = GSemi(train_l, train_f, np.array(test_l), np.array(test_f), np.array(test_id), ext_f)
        # model.fit(from_fed, to_fed, file_path, 16)
        # model = DecisionTreeSemi(train_l, train_f, np.array(test_l), np.array(test_f), np.array(test_id), ext_f)
        # model.fit(FED_NUM, "/data/icde_data")
        train_l = np.concatenate((train_l, ext_l))
        train_f = np.concatenate((train_f, ext_f))
        model = LinearRegression(
            train_l,
            train_f,
            np.array(test_l),
            np.array(test_f),
            np.array(test_id))
        model.fit(FED_NUM, 8, file_path)
