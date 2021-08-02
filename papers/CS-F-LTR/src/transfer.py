"""[summary]
"""
import getopt
import sys
import random
import numpy as np
from sklearn import preprocessing
from src.data_preprocess import load_only, split_test_train
from src.mapper import Mapper

TOP_NUM = 5000  # 1000000
FED_NUM = 3  # 8
EPSILON = 0.5
IS_SEMI = True
FEATURE_NUM = 51
POS_NUM = 20
with_sketch = 1
D = 5
M = 60

IS_COMMAND = True

opts, args = getopt.gnu_getopt(sys.argv[1:], 'e:w:d:m:h', [
    'eps=', 'with_sketch=', 'dim=', 'wid=', 'help'])
for opt_name, opt_val in opts:
    if opt_name in ("-e", "--eps"):
        EPSILON = float(opt_val)
    elif opt_name in ("-w", "--with_sketch"):
        with_sketch = int(opt_val)
    elif opt_name in ('-d', "--dim"):
        D = int(opt_val)
    elif opt_name in ('-m', "--wid"):
        M = int(opt_val)

DATA_PATH = "./data/fed_std_w%d_d%d_m%d_e%.5f_%d/" % (
    with_sketch, D, M, EPSILON, TOP_NUM)

all_train_features = []
all_train_query_ids = []
all_train_relevance_labels = []

all_test_features = []
all_test_query_ids = []
all_test_relevance_labels = []

all_train_features_u = []
all_train_query_ids_u = []

mapper = Mapper()
mapper = mapper.load_mapper("./data/", 'mapper' + str(TOP_NUM))
all_queries = mapper.get_queries()
query_seg = len(all_queries) // FED_NUM
del mapper

for i in range(FED_NUM):
    raw_cases, raw_labels, raw_ids, ext_cases, ext_labels, ext_ids = load_only(
        DATA_PATH + "fed_std%d.txt" % i, is_semi=IS_SEMI)
    for j in range(len(raw_labels)):
        if raw_labels[j] > 1.5:
            raw_labels[j] = 2
    scaler = preprocessing.StandardScaler().fit(raw_cases)
    train_features = scaler.transform(raw_cases)
    if bool(ext_cases):
        scaler = preprocessing.StandardScaler().fit(ext_cases)
        ext_cases = scaler.transform(ext_cases)
    train_relevance_labels = np.array(raw_labels)
    train_query_ids = np.array(raw_ids)
    train_features, train_query_ids, train_relevance_labels, test_features, test_query_ids, test_relevance_labels = \
        split_test_train(
            train_features,
            train_query_ids,
            train_relevance_labels,
            test_size=0.2,
            random_state=0)
    train_features = list(train_features)
    train_query_ids = list(train_query_ids)
    train_relevance_labels = list(train_relevance_labels)
    train_features_w = []
    train_query_ids_w = []
    train_relevance_labels_w = []
    f_i_rt = []
    for j in range(len(train_relevance_labels)):
        if train_relevance_labels[j] == 0:
            train_relevance_labels_w.append(train_relevance_labels[j])
            train_query_ids_w.append(train_query_ids[j])
            train_features_w.append(train_features[j])
        else:
            f_i_rt.append([train_relevance_labels[j], train_query_ids[j], train_features[j]])
    random.seed(10)
    f_i_rt = random.sample(f_i_rt, min(POS_NUM, len(f_i_rt)))
    for each in f_i_rt:
        train_relevance_labels_w.append(each[0])
        train_query_ids_w.append(each[1])
        train_features_w.append(each[2])
    train_features = np.array(train_features_w)
    train_query_ids = np.array(train_query_ids_w)
    train_relevance_labels = np.array(train_relevance_labels_w)
    f_i_r = np.concatenate((train_features,
                            train_query_ids.reshape(-1,
                                                    1),
                            train_relevance_labels.reshape(-1,
                                                           1)),
                           axis=1)
    np.random.seed(10)
    np.random.shuffle(f_i_r)
    train_features = list(f_i_r[:, :FEATURE_NUM])
    train_query_ids = list(f_i_r[:, -2])
    train_relevance_labels = list(f_i_r[:, -1])
    ########
    all_train_features.append(np.array(train_features))
    all_train_query_ids.append(np.array(train_query_ids))
    all_train_relevance_labels.append(np.array(train_relevance_labels))
    all_test_features.extend(list(test_features))
    all_test_relevance_labels.extend(list(test_relevance_labels))
    all_test_query_ids.extend(list(test_query_ids))
    all_train_features_u.append(np.array(ext_cases))
    all_train_query_ids_u.append(np.array(ext_ids))

all_test_features = np.array(all_test_features)
all_test_query_ids = np.array(all_test_query_ids)
all_test_relevance_labels = np.array(all_test_relevance_labels)
cnt0, cnt1, cnt2 = 0, 0, 0
for each in all_test_relevance_labels:
    if each < 0.5:
        cnt0 += 1
    elif each < 1.5:
        cnt1 += 1
    else:
        cnt2 += 1
print(cnt0, cnt1, cnt2)

for i in range(FED_NUM):
    c = np.concatenate((all_train_features[i].reshape(-1,
                                                      FEATURE_NUM),
                        all_train_query_ids[i].reshape(-1,
                                                       1),
                        all_train_relevance_labels[i].reshape(-1,
                                                              1)),
                       axis=1)
    np.save(DATA_PATH + 'train_features_ids_labels' + str(i), c)
    c = np.concatenate((all_train_features_u[i].reshape(
        -1, FEATURE_NUM), all_train_query_ids_u[i].reshape(-1, 1)), axis=1)
    np.save(DATA_PATH + 'ext_features_ids' + str(i), c)
c = np.concatenate((all_test_features, all_test_query_ids.reshape(-1, 1),
                    all_test_relevance_labels.reshape(-1, 1)), axis=1)
np.save(DATA_PATH + 'test_features_ids_labels', c)
