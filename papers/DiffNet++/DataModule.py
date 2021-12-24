"""Data Module"""
from __future__ import division
from collections import defaultdict
import numpy as np
#from time import time
#import random

class DataModule():
    """Data Module Class"""
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0

    def Preparemodelsupplement(self):
        """Prepare data"""
        data_dict = {}
        self.Arrangepositivedata()
        self.Generateconsumeditemssparsematrix()
        data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
        data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list

        self.Generateconsumeditemssparsematrixforitemuser()
        data_dict['ITEM_CUSTOMER_INDICES_INPUT'] = self.item_customer_indices_list
        data_dict['ITEM_CUSTOMER_VALUES_INPUT'] = self.item_customer_values_list

        self.Readsocialneighbors()
        self.Generatesocialneighborssparsematrix()
        data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
        data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list
        return data_dict

    def Initializefeaturetrainforuser(self):
        """Initialize train feature for user"""
        self.Readuserfeature()
        self.Arrangepositivefeatureforuser()

    def Initializefeaturetrainforitem(self):
        """Initialize train feature for item"""
        self.Readitemfeature()
        self.Arrangepositivefeatureforitem()

    def Initializerankingtrain(self):
        """Initialize training set"""
        self.Readdata()
        self.Arrangepositivedata()
        self.Arrangepositivedataforitemuser()  # add
        self.Generatetrainnegative()

    def Initializerankingvt(self):
        """Initialize validation set"""
        self.Readdata()
        self.Arrangepositivedata()
        self.Arrangepositivedataforitemuser()  # add
        self.Generatetrainnegative()

    def Initalizerankingeva(self):
        """Initialize test set"""
        self.Readdata()
        self.Getevapositivebatch()
        self.Generateevanegative()

    def Linkedmap(self):
        """Data mapping"""
        self.data_dict['USER_LIST'] = self.user_list
        self.data_dict['ITEM_LIST'] = self.item_list
        self.data_dict['LABEL_LIST'] = self.labels_list
        self.data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
        self.data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
        self.data_dict['ITEM_CUSTOMER_INDICES_INPUT'] = self.item_customer_indices_list
        self.data_dict['ITEM_CUSTOMER_VALUES_INPUT'] = self.item_customer_values_list
        self.data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
        self.data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list

    def Linkedrankingevamap(self):
        """s1 data mapping"""
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list

    def Linkedrankingevamaps4(self):
        """s4 data mapping"""
        self.data_dict['s4_EVA_USER_LIST'] = self.s4_eva_user_list
        self.data_dict['s4_EVA_ITEM_LIST'] = self.s4_eva_item_list

###########################################  Ranking ############################################

    def Readuserfeature(self):
        """Read user feature data"""
        f = open(self.filename)
        total_user_feature_list = set()
        user_feature_hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split(" ")
            user_feature_hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_feature_list.add(int(arr[0]))
        self.user_feature_hash_data = user_feature_hash_data
        self.total_user_feature_list = list(total_user_feature_list)

    def Readuserfeatures4(self):
        """Read user feature data s4"""
        f = open(self.filename)
        total_user_feature_list = set()
        user_feature_hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split(" ")
            user_feature_hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_feature_list.add(int(arr[0]))
        self.s4_user_feature_hash_data = user_feature_hash_data
        self.s4_total_user_feature_list = list(total_user_feature_list)

    def Readitemfeature(self):
        """Read item feature data"""
        f = open(self.filename)
        total_item_feature_list = set()
        item_feature_hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split(" ")
            item_feature_hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_item_feature_list.add(int(arr[0]))
        self.total_item_feature_list = list(total_item_feature_list)
        self.item_feature_hash_data = item_feature_hash_data

    def Readitemfeatures4(self):
        """Read item feature data s4"""
        f = open(self.filename)
        total_item_feature_list = set()
        item_feature_hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split(" ")
            item_feature_hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_item_feature_list.add(int(arr[0]))
        self.s4_total_item_feature_list = list(total_item_feature_list)
        self.s4_item_feature_hash_data = item_feature_hash_data

    def Readdata(self):
        """Read data"""
        f = open(self.filename)
        total_user_list = set()
        hash_data = defaultdict(int)  # hash_data = 0
        for i, line in enumerate(f):
            if i == 0:
                continue
            arr = line.split(" ")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data

    def Arrangepositivedata(self):
        """Arrange positive data"""
        positive_data = defaultdict(set)
        user_item_num_dict = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)

        for u in range(self.conf.num_users):
            user_item_num_dict[u] = len(positive_data[u])+1
        self.positive_data = positive_data
        self.user_item_num_dict = user_item_num_dict
        self.total_data = len(total_data)


    def Arrangepositivefeatureforuser(self):
        """Arrange positive feature for user"""
        positive_user_feature = defaultdict(set)
        user_feature_num_dict = defaultdict(set)
        total_user_feature = set()
        user_feature_hash_data = self.user_feature_hash_data
        for (u, i) in user_feature_hash_data:
            total_user_feature.add((u, i))
            positive_user_feature[u].add(i)

        for u in positive_user_feature:
            user_feature_num_dict[u] = len(positive_user_feature[u])
        self.positive_user_feature = positive_user_feature
        self.total_user_feature = len(total_user_feature)
        self.user_feature_num_dict = user_feature_num_dict

    def Arrangepositivefeatureforitem(self):
        """Arrange positive feature for item"""
        positive_item_feature = defaultdict(set)
        item_feature_num_dict = defaultdict(set)
        total_item_feature = set()
        item_feature_hash_data = self.item_feature_hash_data
        for (u, i) in item_feature_hash_data:
            total_item_feature.add((u, i))
            positive_item_feature[u].add(i)

        for u in positive_item_feature:
            item_feature_num_dict[u] = len(positive_item_feature[u])
        self.positive_item_feature = positive_item_feature
        self.total_item_feature = len(total_item_feature)
        self.item_feature_num_dict = item_feature_num_dict

    def Arrangepositivedataforitemuser(self):
        """Arrange positive feature for item user"""
        positive_data_for_item_user = defaultdict(set)
        item_user_num_dict = defaultdict(set)

        total_data_for_item_user = set()
        hash_data_for_item_user = self.hash_data
        for (u, i) in hash_data_for_item_user:
            total_data_for_item_user.add((i, u))
            positive_data_for_item_user[i].add(u)


        for i in range(self.conf.num_items):
            item_user_num_dict[i] = len(positive_data_for_item_user[i]) + 1

        self.item_user_num_dict = item_user_num_dict
        self.positive_data_for_item_user = positive_data_for_item_user
        self.total_data_for_item_user = len(total_data_for_item_user)


    def Generatetrainnegative(self):
        """Generate training negative samples"""
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        self.negative_data = negative_data
        self.terminal_flag = 1

    def Getvtrankingonebatch(self):
        """Get batch data for training or testing"""
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        user_list = []
        item_list = []
        labels_list = []
        for u in total_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(positive_data[u])
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(negative_data[u])
            labels_list.extend([0] * len(negative_data[u]))

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])

    def Gettrainrankingbatch(self):
        """Get batch data for training or testing"""
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        index = self.index
        batch_size = self.conf.training_batch_size

        user_list, item_list, labels_list = [], [], []

        if index + batch_size < len(total_user_list):
            target_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_user_list = total_user_list[index:len(total_user_list)]
            self.index = 0
            self.terminal_flag = 0

        for u in target_user_list:
            user_list.extend([u] * len(positive_data[u]))
            item_list.extend(list(positive_data[u]))
            labels_list.extend([1] * len(positive_data[u]))
            user_list.extend([u] * len(negative_data[u]))
            item_list.extend(list(negative_data[u]))
            labels_list.extend([0] * len(negative_data[u]))

        self.user_list = np.reshape(user_list, [-1, 1])
        self.item_list = np.reshape(item_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])

    def Getevapositivebatch(self):
        """Get test positive batch data"""
        hash_data = self.hash_data
        user_list = []
        item_list = []
        index_dict = defaultdict(list)
        index = 0
        for (u, i) in hash_data:
            user_list.append(u)
            item_list.append(i)
            index_dict[u].append(index)
            index = index + 1
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        self.eva_index_dict = index_dict

    def Generateevanegative(self):
        """Generate test negative samples"""
        hash_data = self.hash_data
        total_user_list = self.total_user_list
        num_evaluate = self.conf.num_evaluate
        num_items = self.conf.num_items
        eva_negative_data = defaultdict(list)
        for u in total_user_list:
            for _ in range(num_evaluate):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                eva_negative_data[u].append(j)
        self.eva_negative_data = eva_negative_data

    def Getevarankingbatch(self):
        """Get Eva batch data"""
        batch_size = self.conf.evaluate_batch_size
        num_evaluate = self.conf.num_evaluate
        eva_negative_data = self.eva_negative_data
        total_user_list = self.total_user_list
        index = self.index
        terminal_flag = 1
        total_users = len(total_user_list)
        user_list = []
        item_list = []
        if index + batch_size < total_users:
            batch_user_list = total_user_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_user_list = total_user_list[index:total_users]
            self.index = 0
        for u in batch_user_list:
            user_list.extend([u]*num_evaluate)
            item_list.extend(eva_negative_data[u])
        self.eva_user_list = np.reshape(user_list, [-1, 1])
        self.eva_item_list = np.reshape(item_list, [-1, 1])
        return batch_user_list, terminal_flag

    #Supplement for Sparse Computation
    def Readsocialneighbors(self, friends_flag=1):
        """Read social neighbors"""
        social_neighbors = defaultdict(set)
        social_neighbors_num_dict = defaultdict(int)

        links_file = open(self.conf.links_filename)
        for _, line in enumerate(links_file):
            tmp = line.split(' ')
            u1, u2 = int(tmp[0]), int(tmp[1])
            social_neighbors[u1].add(u2)
            if friends_flag == 1:
                social_neighbors[u2].add(u1)


        for u in range(self.conf.num_users):
            social_neighbors_num_dict[u] = len(social_neighbors[u])+1

        self.social_neighbors_num_dict = social_neighbors_num_dict
        self.social_neighbors = social_neighbors

    def Generatesocialneighborssparsematrix(self):
        """Generate Social Neighbors Sparse Matrix Indices and Values"""
        social_neighbors = self.social_neighbors
        social_neighbors_num_dict = self.social_neighbors_num_dict  # weight avg

        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        social_neighbors_values_weight_avg_list = []
        social_neighbor_num_list = []
        social_neighbors_dict = defaultdict(list)

        user_user_num_for_sparsity_dict = defaultdict(set)
        user_user_sparsity_dict = {}

        user_user_sparsity_dict['0-4'] = []
        user_user_sparsity_dict['4-8'] = []
        user_user_sparsity_dict['8-16'] = []
        user_user_sparsity_dict['16-32'] = []
        user_user_sparsity_dict['32-64'] = []
        user_user_sparsity_dict['64-'] = []

        for u in range(self.conf.num_users):
            user_user_num_for_sparsity_dict[u] = len(social_neighbors[u])

        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u])

        user_list = sorted(list(social_neighbors.keys()))

        for user in range(self.conf.num_users):
            if user in social_neighbors_dict:
                social_neighbor_num_list.append(
                    len(social_neighbors_dict[user]))
            else:
                social_neighbor_num_list.append(1)

        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_values_list.append(
                    1.0/len(social_neighbors_dict[user]))
                social_neighbors_values_weight_avg_list.append(
                    1.0/(np.sqrt(social_neighbors_num_dict[user])* \
                    np.sqrt(social_neighbors_num_dict[friend])))  # weight avg

        for u in range(self.conf.num_users):
            cur_user_neighbors_num = user_user_num_for_sparsity_dict[u]
            if (cur_user_neighbors_num >= 0) & (cur_user_neighbors_num < 4):
                user_user_sparsity_dict['0-4'].append(u)
            elif (cur_user_neighbors_num >= 4) & (cur_user_neighbors_num < 8):
                user_user_sparsity_dict['4-8'].append(u)
            elif (cur_user_neighbors_num >= 8) & (cur_user_neighbors_num < 16):
                user_user_sparsity_dict['8-16'].append(u)
            elif (cur_user_neighbors_num >= 16) & (cur_user_neighbors_num < 32):
                user_user_sparsity_dict['16-32'].append(u)
            elif (cur_user_neighbors_num >= 32) & (cur_user_neighbors_num < 64):
                user_user_sparsity_dict['32-64'].append(u)
            elif cur_user_neighbors_num >= 64:
                user_user_sparsity_dict['64-'].append(u)

        self.user_user_sparsity_dict = user_user_sparsity_dict

        self.social_neighbors_indices_list = np.array(
            social_neighbors_indices_list).astype(np.int64)
        self.social_neighbors_values_list = np.array(
            social_neighbors_values_list).astype(np.float32)
        self.social_neighbors_values_weight_avg_list = np.array(
            social_neighbors_values_weight_avg_list).astype(np.float32)   # weight avg
        self.social_neighbor_num_list = np.array(
            social_neighbor_num_list).astype(np.int64)

    def Generateconsumeditemssparsematrix(self):
        """Generate Consumed Items Sparse Matrix Indices and Values"""
        positive_data = self.positive_data
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_item_num_list = []
        consumed_items_dict = defaultdict(list)
        user_item_num_for_sparsity_dict = defaultdict(set)
        user_item_sparsity_dict = {}
        user_item_sparsity_dict['0-4'] = []
        user_item_sparsity_dict['4-8'] = []
        user_item_sparsity_dict['8-16'] = []
        user_item_sparsity_dict['16-32'] = []
        user_item_sparsity_dict['32-64'] = []
        user_item_sparsity_dict['64-'] = []

        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])

        user_list = sorted(list(positive_data.keys()))

        for u in range(self.conf.num_users):
            user_item_num_for_sparsity_dict[u] = len(positive_data[u])

        for user in range(self.conf.num_users):
            if user in consumed_items_dict:
                consumed_item_num_list.append(len(consumed_items_dict[user]))
            else:
                consumed_item_num_list.append(1)

        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(
                    1.0/len(consumed_items_dict[u]))

        for u in range(self.conf.num_users):
            cur_user_consumed_item_num = user_item_num_for_sparsity_dict[u]
            if (cur_user_consumed_item_num >= 0) & (cur_user_consumed_item_num < 4):
                user_item_sparsity_dict['0-4'].append(u)
            elif (cur_user_consumed_item_num >= 4) & (cur_user_consumed_item_num < 8):
                user_item_sparsity_dict['4-8'].append(u)
            elif (cur_user_consumed_item_num >= 8) & (cur_user_consumed_item_num < 16):
                user_item_sparsity_dict['8-16'].append(u)
            elif (cur_user_consumed_item_num >= 16) & (cur_user_consumed_item_num < 32):
                user_item_sparsity_dict['16-32'].append(u)
            elif (cur_user_consumed_item_num >= 32) & (cur_user_consumed_item_num < 64):
                user_item_sparsity_dict['32-64'].append(u)
            elif cur_user_consumed_item_num >= 64:
                user_item_sparsity_dict['64-'].append(u)

        self.user_item_sparsity_dict = user_item_sparsity_dict
        self.consumed_items_indices_list = np.array(
            consumed_items_indices_list).astype(np.int64)
        self.consumed_items_values_list = np.array(
            consumed_items_values_list).astype(np.float32)
        self.consumed_item_num_list = np.array(
            consumed_item_num_list).astype(np.int64)

    def Generateconsumeditemssparsematrixforitemuser(self):
        """Generate Consumed Items Sparse Matrix Indices and Values for item user"""
        positive_data_for_item_user = self.positive_data_for_item_user
        item_customer_indices_list = []
        item_customer_values_list = []
        item_customer_num_list = []
        item_customer_dict = defaultdict(list)

        for i in positive_data_for_item_user:
            item_customer_dict[i] = sorted(positive_data_for_item_user[i])
        item_list = sorted(list(positive_data_for_item_user.keys()))

        for item in range(self.conf.num_items):
            if item in item_customer_dict:
                item_customer_num_list.append(len(item_customer_dict[item]))
            else:
                item_customer_num_list.append(1)

        for i in item_list:
            for u in item_customer_dict[i]:
                item_customer_indices_list.append([i, u])
                item_customer_values_list.append(
                    1.0/len(item_customer_dict[i]))

        self.item_customer_indices_list = np.array(
            item_customer_indices_list).astype(np.int64)
        self.item_customer_values_list = np.array(
            item_customer_values_list).astype(np.float32)
        self.item_customer_num_list = np.array(
            item_customer_num_list).astype(np.int64)

    def Generatefeaturesparsematrixforuser(self):
        """Generate feature sparse matrix for user"""
        positive_feature_for_user = self.positive_user_feature
        user_feature_indices_list = []
        user_feature_values_list = []
        #user_feature_values_weight_avg_list = []
        user_feature_num_list = []
        user_feature_dict = defaultdict(list)
        #user_feature_num_dict = self.user_feature_num_dict  # weight avg
        for i in positive_feature_for_user:
            user_feature_dict[i] = sorted(positive_feature_for_user[i])
        user_list = sorted(list(positive_feature_for_user.keys()))

        for u in user_list:
            for f in user_feature_dict[u]:
                user_feature_indices_list.append([u, f])
                user_feature_values_list.append(1.0/len(user_feature_dict[u]))

        self.user_feature_indices_list = np.array(
            user_feature_indices_list).astype(np.int64)
        self.cur_user_feature_user_num = len(user_feature_dict)
        self.user_feature_values_list = np.array(
            user_feature_values_list).astype(np.float32)
        self.user_feature_num_list = np.array(
            user_feature_num_list).astype(np.int64)

    def Generatefeaturesparsematrixforitem(self):
        """Generate feature sparse matrix for item"""
        positive_feature_for_item = self.positive_item_feature
        item_feature_indices_list = []
        item_feature_values_list = []
        #item_feature_values_weight_avg_list = []
        item_feature_num_list = []
        item_feature_dict = defaultdict(list)
        #item_feature_num_dict = self.item_feature_num_dict  # weight avg

        for i in positive_feature_for_item:
            item_feature_dict[i] = sorted(positive_feature_for_item[i])
        item_list = sorted(list(positive_feature_for_item.keys()))

        for v in item_list:
            for f in item_feature_dict[v]:
                item_feature_indices_list.append([v, f])
                item_feature_values_list.append(1.0/len(item_feature_dict[v]))

        self.item_feature_indices_list = np.array(
            item_feature_indices_list).astype(np.int64)
        self.cur_item_feature_item_num = len(item_feature_dict)
        self.item_feature_values_list = np.array(
            item_feature_values_list).astype(np.float32)
        self.item_feature_num_list = np.array(
            item_feature_num_list).astype(np.int64)
