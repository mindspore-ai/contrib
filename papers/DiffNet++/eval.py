"""eval"""
import os
from time import time
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net
from mindspore import context
import mindspore.common.dtype as mstype
import mindspore.numpy as dsnp
import Metrics
import datahelper as dh
from Parserconf import Parserconf
from DataModule import DataModule
from train import Diffnetplus

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

RATING_TRAIN_FILE = "./data/yelp/yelp.train.rating"
RATING_TEST_FILE = "./data/yelp/yelp.test.rating"
RATING_VAL_FILE = "./data/yelp/yelp.val.rating"
TEST = "./data/yelp/test"
LINK_TRAIN_FILE = "../data/yelp/yelp.links"
USER_NUM = 17237
ITEM_NUM = 38342
USER_KEY = 'user_id_col'
ITEM_KEY = 'item_id_col'
LABEL_KEY = 'label_id_col'

config_path = os.path.join(os.getcwd(), 'conf/yelp_Diffnetplus.ini')
print(config_path)
conf = Parserconf(config_path)
conf.Parserconfi()

print("Loading data......")
t1 = time()
train_data = DataModule(conf, RATING_TRAIN_FILE)
train_data.Initializerankingtrain()
train_data_dict = train_data.Preparemodelsupplement()
print('Time of loading data: %ds' %(time() - t1))


tt1 = time()
diffnet_plus = Diffnetplus(conf, train_data_dict)
param_dict = load_checkpoint("./check_point/Diffnetplus_220.ckpt")
load_param_into_net(diffnet_plus, param_dict)

ds_rating_data, train_user_item_dict, train_user_list, train_item_list = \
    dh.load_data(RATING_TRAIN_FILE)
ds_eval_data, eval_user_item_dict, eval_user_list, eval_item_list = \
    dh.load_data(RATING_TEST_FILE)  #user_item_dict 有user_key没过滤

#   Fliter the key values in the eval_user_item_dict
#   which are not available in the eval_user_list

for user_id_check in eval_user_item_dict:
    if user_id_check not in eval_user_list:
        del eval_user_item_dict[user_id_check]
#  —————————— Recond the index of user in eval pos ————————————

index = 0
index_pos_user = {}
for eval_user_id in eval_user_list:
    cur_user = eval_user_id
    index_pos_user[eval_user_id] = []
    cur_pos_user_item_list = eval_user_item_dict[eval_user_id]
    l_cur_pos_user_item_list = len(cur_pos_user_item_list)
    for item in range(l_cur_pos_user_item_list):
        cur_item = cur_pos_user_item_list[item]
        index_pos_user[eval_user_id].append(index)
        index = index + 1

#  —————————— Create negative samples dict for testing ——————————
user_item_neg_dict = np.load('user_item_neg_dict.npy', allow_pickle=True).item()

#  ——————————————  Evaluation ———————————————
eval_batch_user = 500

def format_eval_input_for_neg(user_item_dict, neg_user_list):
    """Eval input data for negative samples"""
    user_list = []
    item_list = []
    label_list = []
    for user_id in neg_user_list:
        cur_item_list = user_item_dict[user_id]
        l_cur_item_list = len(cur_item_list)
        for i in range(l_cur_item_list):
            user_list.append(user_id)
            item_list.append(cur_item_list[i])
            label_list.append(0)
    return dsnp.reshape(Tensor(user_list), (-1, 1, 1)), dsnp.reshape(Tensor(item_list), \
        (-1, 1, 1)), dsnp.reshape(Tensor(label_list), (-1, 1, 1))

#  ——————————————  pos and neg output ————————————————
t4 = time()
pos_prediction_list = diffnet_plus(dsnp.reshape(\
    Tensor(ds_eval_data[USER_KEY], mstype.int64), (-1, 1, 1)), dsnp.reshape(\
    Tensor(ds_eval_data[ITEM_KEY], mstype.int64), (-1, 1, 1)), dsnp.reshape(\
        Tensor(ds_eval_data[LABEL_KEY], mstype.float32), (-1, 1, 1)))
eval_neg_user_list, eval_neg_item_list, eval_neg_label_list = format_eval_input_for_neg(\
    user_item_neg_dict, eval_user_list)

neg_prediction = diffnet_plus(eval_neg_user_list, eval_neg_item_list, eval_neg_label_list)
neg_prediction_dict = dsnp.reshape(neg_prediction, (-1, 1000))
eval_batch_user = 500
print('t4: %.3fmins'%((time() - t4) / 60))

def get_hr_ndcg(index_dict, pos_prediction_input, neg_prediction_input, topk):
    """Get Hit HR and NDCG results"""
    hr_list = []
    ndcg_list = []
    for idx in range(len(eval_user_list)):
        user = eval_user_list[idx]
        cur_user_pos_prediction = pos_prediction_input.asnumpy()[index_dict[user]]
        cur_user_neg_prediction = neg_prediction_input.asnumpy()[idx]

        positive_length = len(cur_user_pos_prediction)
        target_length = min(topk, positive_length)
        total_prediction = np.concatenate([cur_user_pos_prediction, cur_user_neg_prediction])
        # small —> large
        sort_index = np.argsort(total_prediction)
        # large —> small
        sort_index = sort_index[::-1]
        user_hr_list = []
        user_ndcg_list = []
        for i in range(topk):
            ranking = sort_index[i]
            if ranking < positive_length:
                user_hr_list.append(Metrics.Gethr())
                user_ndcg_list.append(Metrics.Getdcg(i))
        idcg = Metrics.Getidcg(target_length)
        tmp_hr = np.sum(user_hr_list) / target_length
        tmp_ndcg = np.sum(user_ndcg_list) / idcg
        hr_list.append(tmp_hr)
        ndcg_list.append(tmp_ndcg)

    return np.mean(hr_list), np.mean(ndcg_list)

print('getting hr and ndcg.........')
t5 = time()
print('top_5')
hr_5, ndcg_5 = get_hr_ndcg(index_pos_user, pos_prediction_list, neg_prediction_dict, 5)
print('top_10')
hr_10, ndcg_10 = get_hr_ndcg(index_pos_user, pos_prediction_list, neg_prediction_dict, 10)
print('top_15')
hr_15, ndcg_15 = get_hr_ndcg(index_pos_user, pos_prediction_list, neg_prediction_dict, 15)
print('top_20')
hr_20, ndcg_20 = get_hr_ndcg(index_pos_user, pos_prediction_list, neg_prediction_dict, 20)
print('Time of getting hr and ndcg: %.3fmins'%((time() - t5) / 60))

print("hr_5: %.3f, ndcg_5: %.3f"%(hr_5, ndcg_5))
print("hr_10: %.3f, ndcg_10: %.3f"%(hr_10, ndcg_10))
print("hr_15: %.3f, ndcg_15: %.3f"%(hr_15, ndcg_15))
print("hr_20: %.3f, ndcg_20: %.3f"%(hr_20, ndcg_20))

print('Evaluation time: %.2fmins' % ((time() - tt1) / 60))
