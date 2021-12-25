"""Data Helper"""
#import numpy as np
import random as rd
import pandas as pd

USER_NUM = 17237
ITEM_NUM = 38342
USER_KEY = 'user_id_col'
ITEM_KEY = 'item_id_col'
LABEL_KEY = 'label_id_col'

def load_data(file_path):
    """load raw data"""
    data = {}
    with open(file_path) as f:
        df = pd.read_csv(f, sep=' ')
        df.columns = ['user_id', 'item_id', 'label']
    data[USER_KEY] = df['user_id'].values
    data[ITEM_KEY] = df['item_id'].values
    data[LABEL_KEY] = df['label'].values

    user_list = sorted(df['user_id'].unique())
    item_list = sorted(df['item_id'].unique())

    # cur tuple list
    #data_record = []
    user_item_dict = {}
    l_data = len(data[USER_KEY])

    for idx in range(l_data):
        cur_user = data[USER_KEY][idx]
        cur_item = data[ITEM_KEY][idx]
        #cur_label = data[LABEL_KEY][idx]
        if cur_user in user_item_dict:
            temp = user_item_dict[cur_user]
            if cur_item not in temp:
                temp.append(cur_item)
        else:
            temp = []
            temp.append(cur_item)
        user_item_dict[cur_user] = temp

    return data, user_item_dict, user_list, item_list

def get_val_data_from_source(record_dict):
    """Get validation data as dict"""
    user_list = sorted(record_dict.keys())
    t_user_list = []
    t_item_list = []
    t_label_list = []

    for user_idx in user_list:
        cur_user = user_idx
        cur_pos_item_list = record_dict[user_idx]
        l_cur_pos_item_list = len(cur_pos_item_list)
        for i in range(l_cur_pos_item_list):
            t_user_list.append(cur_user)
            t_item_list.append(cur_pos_item_list[i])
            t_label_list.append(1)
    validation_dict = {}
    validation_dict[USER_KEY] = t_user_list
    validation_dict[ITEM_KEY] = t_item_list
    validation_dict[LABEL_KEY] = t_label_list

    return validation_dict

def get_eval_data_from_source(record_dict):
    """Get eva data as dict"""
    user_list = sorted(record_dict.keys())
    t_user_list = []
    t_item_list = []
    t_label_list = []

    for user_idx in user_list:
        cur_user = user_idx
        cur_pos_item_list = record_dict[user_idx]
        l_cur_pos_item_list = len(cur_pos_item_list)
        for i in range(l_cur_pos_item_list):
            t_user_list.append(cur_user)
            t_item_list.append(cur_pos_item_list[i])
            t_label_list.append(1)
    eval_dict = {}
    eval_dict[USER_KEY] = t_user_list
    eval_dict[ITEM_KEY] = t_item_list
    eval_dict[LABEL_KEY] = t_label_list

    return eval_dict

def get_train_data_from_source(record_dict, neg_num=2):
    """Get train data as dict"""
    user_list = sorted(record_dict.keys())
    neg_user_item_dict = {}

    for user_idx in user_list:
        cur_user = user_idx
        cur_pos_item_list = record_dict[user_idx]
        if cur_user not in neg_user_item_dict:
            neg_user_item_dict[cur_user] = []
        for i in range(neg_num):
            item_id = rd.randint(1, ITEM_NUM)
            while item_id in cur_pos_item_list + neg_user_item_dict[cur_user]:
                item_id = rd.randint(1, ITEM_NUM)
            neg_user_item_dict[cur_user].append(item_id)
        #neg_user_item_dict[user_idx] = cur_neg_item_list
    t_user_list = []
    t_item_list = []
    t_label_list = []

    for user_idx in user_list:
        cur_user = user_idx
        cur_pos_item_list = record_dict[user_idx]
        cur_neg_item_list = neg_user_item_dict[user_idx]
        l_cur_pos_item_list = len(cur_pos_item_list)
        l_cur_neg_item_list = len(cur_neg_item_list)
        for i in range(l_cur_pos_item_list):
            t_user_list.append(cur_user)
            t_item_list.append(cur_pos_item_list[i])
            t_label_list.append(1.0)
        for i in range(l_cur_neg_item_list):
            t_user_list.append(cur_user)
            t_item_list.append(cur_neg_item_list[i])
            t_label_list.append(0.0)
    s_neg_dict = {}
    s_neg_dict[USER_KEY] = t_user_list
    s_neg_dict[ITEM_KEY] = t_item_list
    s_neg_dict[LABEL_KEY] = t_label_list

    return record_dict, s_neg_dict
