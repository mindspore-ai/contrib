"""
This data implement model's dataloader

:paper author: hxq
:code author: hxq
:code convert: shy
"""

import random
import os
import csv
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

ATTR_ID = 2
#num_group = 6
def load_data(data_dir):
    """
    @params:
        data_dir: data folder
    @returns:
        data
    """
    if 'cite' in data_dir:
        user_item_matrix, train_gender_dict = citeulike(data_dir, tag_occurence_thres=10)
        train_data, vad_data, tst_data = split_data(user_item_matrix, split_ratio=(3, 1, 1))
        train_data = train_data.tocsr()
        vad_data = vad_data.tocsr()
        tst_data = tst_data.tocsr()
        n_users = user_item_matrix.shape[0]
        n_items = user_item_matrix.shape[1]
        vad_gender_dict = train_gender_dict
        tst_gender_dict = train_gender_dict

    else:#'ml', 'lastfm', 'netflix' in data_dir:

        pro_dir = os.path.join(data_dir, 'pro_sg')

        unique_sid = []
        with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)

        train_data, train_gender_dict, vad_data, \
        vad_gender_dict, tst_data, tst_gender_dict, n_users = \
        load_train_vad_test_data(os.path.join(pro_dir, 'train.csv'),
                                 os.path.join(pro_dir, 'validation.csv'),
                                 os.path.join(pro_dir, 'test.csv'),
                                 n_items)

    assert n_items == train_data.shape[1]
    assert n_items == vad_data.shape[1]
    assert n_items == tst_data.shape[1]

    return (n_items, n_users, train_data, train_gender_dict, vad_data, vad_gender_dict,
            tst_data, tst_gender_dict)

def load_train_vad_test_data(train_csv_file, vad_csv_file, test_csv_file, n_items):
    """
    @params:
        train_csv_file,
        vad_csv_file,
        test_csv_file,
        n_items
    @returns:
        data
    """
    tp_train = pd.read_csv(train_csv_file)
    n_users = tp_train['uid'].max() + 1

    rows, cols = tp_train['uid'], tp_train['sid']
    num_group = tp_train['gender_age'].max() + 1

    data_train = sp.csr_matrix((np.ones_like(rows), (rows, cols)),
                               dtype='float32', shape=(n_users, n_items))

    # gender_dict = dict((row[0], row[2]) for index, row in tp.iterrows())
    group_dict_train = dict((i, set()) for i in range(num_group))
    for index, row in tp_train.iterrows():
        print(index)
        group_dict_train[int(row[ATTR_ID])].add(row[0])
    for group in group_dict_train:
        group_dict_train[group] = list(group_dict_train[group])

    tp_vad = pd.read_csv(vad_csv_file)
    rows, cols = tp_vad['uid'], tp_vad['sid']
    data_vad = sp.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float32',
                             shape=(n_users, n_items))
    group_dict_vad = dict((i, set()) for i in range(num_group))
    for index, row in tp_vad.iterrows():
        group_dict_vad[int(row[ATTR_ID])].add(row[0])
    for group in group_dict_vad:
        group_dict_vad[group] = list(group_dict_vad[group])


    tp_test = pd.read_csv(test_csv_file)
    rows, cols = tp_test['uid'], tp_test['sid']
    data_test = sp.csr_matrix((np.ones_like(rows), (rows, cols)),
                              dtype='float32', shape=(n_users, n_items))
    group_dict_test = dict((i, set()) for i in range(num_group))
    for index, row in tp_test.iterrows():
        group_dict_test[int(row[ATTR_ID])].add(row[0])
    for group in group_dict_test:
        group_dict_test[group] = list(group_dict_test[group])


    return data_train, group_dict_train, data_vad, group_dict_vad, \
           data_test, group_dict_test, n_users



def load_train_data(csv_file, n_items):
    """
    @params:
        csv_file,
        n_items
    @returns:
        data
    """
    train_data = pd.read_csv(csv_file)
    n_users = train_data['uid'].max() + 1
    rows, cols = train_data['uid'], train_data['sid']
    num_group = train_data['gender_age'].max() + 1
    data = sp.csr_matrix((np.ones_like(rows), (rows, cols)),
                         dtype='float32', shape=(n_users, n_items))
    group_dict = dict((i, set()) for i in range(num_group))
    for index, row in train_data.iterrows():
        print(index)
        group_dict[int(row[ATTR_ID])].add(row[0])
    for group in group_dict:
        group_dict[group] = list(group_dict[group])
    return data, group_dict, n_users


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    """
    @params:
        csv_file_tr,
        csv_file_te,
        n_items
    @returns:
        data
    """
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)
    num_group = tp_tr['gender_age'].max() + 1

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']  # pandas.Series类型
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    group_dict = dict((i, set()) for i in range(num_group))
    for index, row in tp_tr.iterrows():
        print(index)
        group_dict[row[ATTR_ID]].add(row[0]-start_idx)

    for group in group_dict:
        group_dict[group] = list(group_dict[group])
    # gender_age_tr_dict = dict((row[0]-start_idx, row[ATTR_ID]) for index,row in tp_tr.iterrows())
    # gender_age_te_dict = dict((row[0]-start_idx, row[ATTR_ID]) for index,row in tp_te.iterrows())
    # print('len(gender_age_dict)', len(gender_age_tr_dict))
    # print(gender_te_dict)
    data_tr = sp.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float32',
                            shape=(end_idx - start_idx + 1, n_items))
    data_te = sp.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float32',
                            shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te, group_dict #gender_age_tr_dict, gender_age_te_dict


def load_item_cate(data_dir, num_items):
    """
    @params:
        data_dir,
        n_items
    @returns:
        data
    """
    assert 'alishop' in data_dir
    data_dir = os.path.join(data_dir, 'pro_sg')

    hash_to_sid = {}
    with open(os.path.join(data_dir, 'unique_sid.txt')) as fin:
        for i, line in enumerate(fin):
            hash_to_sid[int(line)] = i
    assert num_items == len(hash_to_sid)

    hash_to_cid = {}
    with open(os.path.join(data_dir, 'item_cate.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for item, cate in reader:
            item, cate = int(item), int(cate)
            if item not in hash_to_sid:
                continue
            assert item in hash_to_sid
            if cate not in hash_to_cid:
                hash_to_cid[cate] = len(hash_to_cid)
    num_cates = len(hash_to_cid)

    item_cate = np.zeros((num_items, num_cates), dtype=np.bool)
    with open(os.path.join(data_dir, 'item_cate.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for item, cate in reader:
            item, cate = int(item), int(cate)
            if item not in hash_to_sid:
                continue
            item = hash_to_sid[item]
            cate = hash_to_cid[cate]
            item_cate[item, cate] = True
    item_cate = item_cate.astype(np.int64)

    joke_sprit = np.argsort(item_cate.sum(axis=0))[-7:]
    item_cate = item_cate[:, joke_sprit]
    assert np.min(np.sum(item_cate, axis=1)) == 1
    assert np.max(np.sum(item_cate, axis=1)) == 1
    return item_cate


def construct_csr_matrix(ids, num):
    """
    @params:
        ids,
        n
    @returns:
        csr_matrix
    """
    rows = np.array(list(range(len(ids))))
    return sp.csr_matrix((np.ones_like(rows), (rows, np.array(ids))), shape=(len(ids), num))


def citeulike(data_dir, tag_occurence_thres=10):
    """
    @params:
        data_dir,
        tag_occurence_thres
    @returns:
        csr_matrix
    """
    user_dict = defaultdict(set)
    group_user_dict = {}
    for i in range(5):
        group_user_dict[i] = set()

    uid = 0
    file_dir = os.path.join(data_dir, 'users.dat')
    for u_index, item_list in enumerate(open(file_dir).readlines()):
        print(u_index)
        items = item_list.strip().split(" ")
        # ignore the first element in each line, which is the number of items the user liked.
        if int(items[0]) > tag_occurence_thres:
            if int(items[0]) <= 20:
                group = 0
            elif int(items[0]) > 20 and int(items[0]) <= 50:
                group = 1
            elif int(items[0]) > 50 and int(items[0]) <= 100:
                group = 2
            elif int(items[0]) > 100 and int(items[0]) <= 200:
                group = 3
            elif int(items[0]) > 200:
                group = 4
            group_user_dict[group].add(uid)


            for item in items[1:]:
                user_dict[uid].add(int(item))
            uid += 1

    for group in group_user_dict:
        group_user_dict[group] = list(group_user_dict[group])
    # group_num = [len(group_user_dict[group]) for group in group_user_dict]
    n_users = len(user_dict)
    n_items = max([item for items in user_dict.values() for item in items]) + 1


    user_item_matrix = sp.dok_matrix((n_users, n_items), dtype=np.int32)
    uid = 0
    for u_index, item_list in enumerate(open(file_dir).readlines()):
        print(u_index)
        items = item_list.strip().split(" ")
        # ignore the first element in each line, which is the number of items the user liked.
        if int(items[0]) > tag_occurence_thres:
            for item in items[1:]:
                user_item_matrix[uid, int(item)] = 1
            uid += 1

    return user_item_matrix, group_user_dict



def split_data(user_item_matrix, split_ratio=(3, 1, 1), seed=1):
    """
    @params:
        user_item_matrix,
        split_ratio,
        seed
    @returns:
        data
    """
    # set the seed to have deterministic results
    np.random.seed(seed)
    train = sp.dok_matrix(user_item_matrix.shape)
    validation = sp.dok_matrix(user_item_matrix.shape)
    test = sp.dok_matrix(user_item_matrix.shape)
    # convert it to lil format for fast row access

    user_item_matrix = sp.lil_matrix(user_item_matrix)
    for user in tqdm(range(user_item_matrix.shape[0]), desc="Split data into train/valid/test"):
        items = list(user_item_matrix.rows[user])
        # items是非零值对应的index
        if len(items) >= 5:

            np.random.shuffle(items)  # # # # #

            train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

            for i in items[0: train_count]:
                train[user, i] = 1
            for i in items[train_count: train_count + valid_count]:
                validation[user, i] = 1
            for i in items[train_count + valid_count:]:
                test[user, i] = 1
    print("{}/{}/{} train/valid/test samples".format(
        len(train.nonzero()[0]),
        len(validation.nonzero()[0]),
        len(test.nonzero()[0])))
    return train, validation, test




def sampler(train_data, st_idx_dict, batch_size, train_group_dict,
            data_te=None, data_vad=None, nu_list=None):
    """
    @params:
        train_data,
        st_idx_dict,
        batch_size,
        train_group_dict,
        data_te,
        data_vad,
        nu_list,
    @returns
    """
    statistics = np.zeros(len(train_group_dict))
    for group in train_group_dict:
        statistics[group] = len(train_group_dict[group])
    statistics = statistics / np.sum(statistics)
    statistics = statistics * batch_size

    indices = []
    x_gender = []

    for group in train_group_dict:
        end_idx = min(st_idx_dict[group]+int(statistics[group]), len(train_group_dict[group]))
        if end_idx-st_idx_dict[group] < 1:
            indices.extend([random.choice(train_group_dict[group])])
            x_gender.extend([group])
        else:
            indices.extend(train_group_dict[group][st_idx_dict[group]:end_idx])
            x_gender.extend([group]*(end_idx - st_idx_dict[group]))

        st_idx_dict[group] = end_idx

    # indices: user_id, 要保留的
    for i, gender in zip(indices, x_gender):
        if np.count_nonzero(train_data[i].toarray()) == 0:
        # if not train_data.rows[i]:
            del indices[i]
            del x_gender[gender]
            print('remove')
    x_input = train_data[indices]

    # print('x.shape', x.shape)
    # print('x_gender.shape', len(x_gender))
    if data_te is None:
        if nu_list is None:
            return x_gender, x_input, st_idx_dict
        nu_list_batch = [nu_list[i] for i in indices]
        return x_gender, x_input, st_idx_dict, nu_list_batch
    if data_vad is None:
        if nu_list is None:
            return x_gender, x_input, data_te[indices], st_idx_dict
        nu_list_batch = [nu_list[i] for i in indices]
        return x_gender, x_input, data_te[indices], st_idx_dict, nu_list_batch
    return x_gender, x_input, data_te[indices], data_vad[indices], st_idx_dict


def sampler_to_list_with_negtive(train_data, st_idx_dict, batch_size,
                                 train_group_dict, num_negatives=4):
    """[summary]

    Args:
        train_data ([type]): [description]
        st_idx_dict ([type]): [description]
        batch_size ([type]): [description]
        train_group_dict ([type]): [description]
        num_negatives (int, optional): [description]. Defaults to 4.

    Returns:
        [type]: [description]
    """
    statistics = np.zeros(len(train_group_dict))
    num_items = train_data.shape[1]
    for group in train_group_dict:
        statistics[group] = len(train_group_dict[group])
    statistics = statistics / np.sum(statistics)
    statistics = statistics * batch_size

    indices = []
    x_gender = []

    for group in train_group_dict:
        end_idx = min(st_idx_dict[group]+int(statistics[group]), len(train_group_dict[group]))
        if end_idx-st_idx_dict[group] < 1:
            indices.extend([random.choice(train_group_dict[group])])
            x_gender.extend([group])
        else:
            indices.extend(train_group_dict[group][st_idx_dict[group]:end_idx])
            x_gender.extend([group]*(end_idx - st_idx_dict[group]))
        st_idx_dict[group] = end_idx

    # indices: user_id, 要保留的
    user_input, item_input, labels, group_ids = [], [], [], []
    for index, user in enumerate(indices):
        #item_tmp = np.nonzero(train_data[u])[0]  #np.nonzero()返回的是tuple, 用[0]选择tuple中的值
        item_tmp = train_data.rows[user]
        for i in item_tmp:
            user_input.append(user)
            item_input.append(i)
            labels.append(1)
            group_ids.append(x_gender[index])
            for t_index in range(num_negatives):
                print(t_index)
                j = np.random.randint(num_items)

                while train_data[user, j] == 1:
                    j = np.random.randint(num_items)
                user_input.append(user)
                item_input.append(j)
                labels.append(0)
                group_ids.append(x_gender[index])

    return user_input, item_input, labels, group_ids
