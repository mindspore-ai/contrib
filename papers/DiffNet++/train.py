"""Train"""
import os
from time import time
from collections import defaultdict
import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import dataset as ds
from mindspore import Tensor, SparseTensor, nn, Model, save_checkpoint, context
from mindspore.nn.layer.activation import get_activation
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.train.callback import Callback
from mindspore.nn.loss.loss import _Loss
from mindspore import ops
import datahelper as dh
from DataModule import DataModule
from Parserconf import Parserconf


def get_embedding_from_neighbors(sparse_tensor, all_embedding):
    """Get embedding from neighbors"""
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    dense_shape = sparse_tensor.dense_shape
    indices = indices.asnumpy()
    values = values.asnumpy()
    all_embedding = all_embedding.asnumpy()
    idxs = defaultdict(list)
    for _, ele in enumerate(indices):
        x, y = ele[0], ele[1]
        idxs[x].append(y)
    out_shape = (dense_shape[0], 16)
    output = np.zeros(shape=out_shape, dtype=np.float32)
    for j in idxs.keys():
        output[j] = np.sum(all_embedding[np.array(idxs[j])], axis=0) / len(idxs[j])
    output = Tensor(output, mstype.float32)
    return output

def softmax(values, indices, row):
    """softmax"""
    indices = indices.asnumpy()
    values = values.asnumpy()
    idxs = defaultdict(list)
    row = row
    for _, ele in enumerate(indices):
        x, y = ele[0], ele[1]
        idxs[x].append(y)
    sum_value = [0] * len(idxs.keys())
    for i in range(len(values)):
        idth = indices[i][0]
        sum_value[idth] += values[i]
    for i in range(len(values)):
        idth = indices[i][0]
        values[i] = np.exp(values[i]) / (sum_value[idth])
    values = Tensor(values, dtype=ms.float32)
    return values

#DATA_DIR
RATING_TRAIN_FILE = "./data/yelp/yelp.train.rating"
RATING_TEST_FILE = "./data/yelp/yelp.test.rating"
RATING_VAL_FILE = "./data/yelp/yelp.val.rating"
TEST = "./data/yelp/test"
LINK_TRAIN_FILE = "../data/yelp/yelp.links"
user_num = 17237
item_num = 38342
USER_KEY = 'user_id_col'
ITEM_KEY = 'item_id_col'
LABEL_KEY = 'label_id_col'

def load_data(file_path):
    """data load"""
    data = {}
    with open(file_path) as f:
        df = pd.read_csv(f, sep=' ')
        df.columns = ['user_id', 'item_id', 'label']
    data['user_id_col'] = df['user_id'].values
    data['item_id_col'] = df['item_id'].values
    data['label_id_col'] = df['label'].values
    user_list = sorted(df['user_id'].unique())
    item_list = sorted(df['item_id'].unique())

    return data, user_list, item_list

class DatasetDiff:
    """The class of reading data"""
    def __init__(self, user_col, item_col, label_col):
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self._num_users = user_num
        self._num_items = item_num

    def __getitem__(self, index):
        user = self.user_col[index]
        label = self.label_col[index]
        item = self.item_col[index]

        user = np.reshape(user, (-1, 1)).astype(np.int32)
        label = np.reshape(label, (-1, 1)).astype(np.float32)
        item = np.reshape(item, (-1, 1)).astype(np.int32)
        return user, item, label

    def __len__(self):
        return len(self.user_col)

def create_dataset(data_source, batch_size=2):
    """Create dataset"""
    samplelist = ds.RandomSampler(num_samples=None, replacement=False)
    input_data = ds.GeneratorDataset(data_source, column_names=['user_id_col', \
        'item_id_col', 'label_id_col'], sampler=samplelist)
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(1)
    return input_data

class Diffnetplus(nn.Cell):
    """The model class of DiffNet++"""
    def __init__(self, conf, data_dict):
        super(Diffnetplus, self).__init__()
        self.conf, self.data_dict = conf, data_dict
        self.d = self.conf.dimension
        self.reduce_sum, self.reshape = P.ReduceSum(), P.Reshape()
        self.exp, self.concat = P.Exp(), ops.Concat(axis=1)
        self.squeeze, self.mul = P.Squeeze(axis=1), P.Mul()
        self.softmax, self.sigmoid = ops.Softmax(), P.Sigmoid()
        self.activation, self.embedding_initializer = get_activation('sigmoid'), 'normal'
        self.embedding_user = nn.Embedding(self.conf.num_users, self.d,\
            embedding_table=self.embedding_initializer)
        self.embedding_item = nn.Embedding(self.conf.num_items, self.d,\
            embedding_table=self.embedding_initializer)
        self.reduce_dimension_layer = nn.Dense(150, self.d, activation=self.activation)
        self.reduce_dimension_layer1 = nn.Dense(\
            self.d * 3, self.d, activation=self.activation)
        self.reduce_dimension_layer2 = nn.Dense(self.d * 3, \
            self.d, activation=self.activation)
        self.logits_dense = nn.Dense(self.d, 1, 'normal', activation=None)
        self.layer1_user_social_attention = nn.Dense(self.d * 2, 1, activation=self.activation)
        self.layer1_user_interest_attention = nn.Dense(self.d * 2, 1, activation=self.activation)
        self.layer1_item_users_attention = nn.Dense(self.d, 1, activation=self.activation)
        self.layer1_item_attention = nn.Dense(self.d, 1, activation=self.activation)
        self.layer2_user_social_attention = nn.Dense(self.d * 2, 1, activation=self.activation)
        self.layer2_user_interest_attention = nn.Dense(self.d * 2, 1, activation=self.activation)
        self.layer2_item_users_attention = nn.Dense(self.d, 1, activation=self.activation)
        self.layer2_item_attention = nn.Dense(self.d, 1, activation=self.activation)
        self.social_neighbors_indices_list = Tensor(\
            self.data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'])
        self.social_neighbors_values_list = Tensor(\
            self.data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'], mstype.float32)
        self.social_neighbors_dense_shape = tuple(np.array(
            [self.conf.num_users, self.conf.num_users]).astype(np.int64))
        self.social_graph_node_attention1 = Tensor(C.random_ops.normal( \
            (len(self.social_neighbors_values_list), 1), \
            mean=Tensor(0.0, mstype.float32), stddev=Tensor(1.0, mstype.float32)))
        self.layer1_social_graph_node_attention = nn.Dense(1, 1, activation=self.activation)
        self.social_neighbors_values_list2 = self.reshape(self.exp( \
            self.layer1_social_graph_node_attention(self.social_graph_node_attention1)), (1, \
            len(self.social_neighbors_values_list)))
        self.social_neighbors_values_list2 = softmax(self.social_neighbors_values_list2, \
            self.social_neighbors_indices_list, self.conf.num_users)
        self.social_graph_node_attention2 = Tensor(C.random_ops.normal(\
            (len(self.social_neighbors_values_list), 1), mean=Tensor(0.0, mstype.float32), \
            stddev=Tensor(1.0, mstype.float32)))
        self.layer2_social_graph_node_attention = nn.Dense(1, 1, activation=self.activation)
        self.social_neighbors_values_list3 = self.reshape(self.exp(
            self.layer2_social_graph_node_attention(self.social_graph_node_attention2)), (1, \
            len(self.social_neighbors_values_list)))
        self.social_neighbors_values_list3 = softmax(self.social_neighbors_values_list3, \
            self.social_neighbors_indices_list, self.conf.num_users)
        self.consumed_items_indices_list = Tensor(self.data_dict['CONSUMED_ITEMS_INDICES_INPUT'])
        self.consumed_items_values_list = Tensor(\
            self.data_dict['CONSUMED_ITEMS_VALUES_INPUT'], mstype.float32)
        self.consumed_items_dense_shape = tuple(np.array(
            [self.conf.num_users, self.conf.num_items]).astype(np.int64))
        self.user_item_node_attention1 = Tensor(C.random_ops.normal(
            (len(self.consumed_items_values_list), 1), mean=Tensor(0.0, mstype.float32), \
            stddev=Tensor(1.0, mstype.float32)))
        self.layer1_user_item_node_attention = nn.Dense(1, 1, activation=self.activation)
        self.consumed_items_values_list2 = self.reshape(self.exp(
            self.layer1_user_item_node_attention(self.user_item_node_attention1)), (1, \
            len(self.consumed_items_values_list)))
        self.consumed_items_values_list2 = softmax(self.consumed_items_values_list2, \
            self.consumed_items_indices_list, self.conf.num_items)
        self.user_item_node_attention2 = Tensor(C.random_ops.normal(
            (len(self.consumed_items_values_list), 1), mean=Tensor(0.0, mstype.float32), \
            stddev=Tensor(1.0, mstype.float32)))
        self.layer2_user_item_node_attention = nn.Dense(1, 1, activation=self.activation)
        self.consumed_items_values_list3 = self.reshape(self.exp(
            self.layer2_user_item_node_attention(self.user_item_node_attention2)), (1, \
            len(self.consumed_items_values_list)))
        self.consumed_items_values_list3 = softmax(self.consumed_items_values_list3, \
            self.consumed_items_indices_list, self.conf.num_items)
        self.item_customer_indices_list = Tensor(self.data_dict['ITEM_CUSTOMER_INDICES_INPUT'])
        self.item_customer_values_list = Tensor(\
            self.data_dict['ITEM_CUSTOMER_VALUES_INPUT'], mstype.float32)
        self.item_customer_dense_shape = tuple(np.array(
            [self.conf.num_items, self.conf.num_users]).astype(np.int64))
        self.item_user_node_attention1 = Tensor(C.random_ops.normal(
            (len(self.item_customer_values_list), 1), mean=Tensor(0.0, mstype.float32), \
            stddev=Tensor(1.0, mstype.float32)))
        self.layer1_item_user_node_attention = nn.Dense(1, 1, activation=self.activation)
        self.item_customer_values_list2 = self.reshape(self.exp(
            self.layer1_item_user_node_attention(self.item_user_node_attention1)), (1, \
            len(self.item_customer_values_list)))
        self.item_customer_values_list2 = softmax(self.item_customer_values_list2, \
            self.item_customer_indices_list, self.conf.num_users)
        self.item_user_node_attention2 = Tensor(C.random_ops.normal(
            (len(self.item_customer_values_list), 1), mean=Tensor(0.0, mstype.float32), \
            stddev=Tensor(1.0, mstype.float32)))
        self.layer2_item_user_node_attention = nn.Dense(1, 1, activation=self.activation)
        self.item_customer_values_list3 = self.reshape(self.exp(
            self.layer2_item_user_node_attention(self.item_user_node_attention2)), (1, \
            len(self.item_customer_values_list)))
        self.item_customer_values_list3 = softmax(self.item_customer_values_list3, \
            self.item_customer_indices_list, self.conf.num_users)
    def construct(self, user_input, item_input, labels):
        """construct function"""
        self.social_neighbors_sparse_matrix = SparseTensor(indices=self.social_neighbors_indices_list,\
            values=self.social_neighbors_values_list, \
            dense_shape=self.social_neighbors_dense_shape)
        self.social_neighbors_sparse_matrix2 = SparseTensor(\
            indices=self.social_neighbors_indices_list, \
            values=self.social_neighbors_values_list2, \
            dense_shape=self.social_neighbors_dense_shape)
        self.social_neighbors_sparse_matrix3 = SparseTensor(\
            indices=self.social_neighbors_indices_list, \
            values=self.social_neighbors_values_list3, \
            dense_shape=self.social_neighbors_dense_shape)
        self.consumed_items_sparse_matrix = SparseTensor(\
            indices=self.consumed_items_indices_list, \
            values=self.consumed_items_values_list, \
            dense_shape=self.consumed_items_dense_shape)
        self.consumed_items_sparse_matrix2 = SparseTensor(\
            indices=self.consumed_items_indices_list, \
            values=self.consumed_items_values_list2, \
            dense_shape=self.consumed_items_dense_shape)
        self.consumed_items_sparse_matrix3 = SparseTensor(\
            indices=self.consumed_items_indices_list, \
            values=self.consumed_items_values_list3, \
            dense_shape=self.consumed_items_dense_shape)
        labels = 1
        self.user_feature_matrix = Tensor(\
            np.load(self.conf.user_review_vector_matrix), mstype.float32)
        self.item_feature_matrix = Tensor(\
            np.load(self.conf.item_review_vector_matrix), mstype.float32)
        self.all_user_embedding = self.embedding_user(Tensor(\
            np.reshape(np.arange(0, self.conf.num_users), (self.conf.num_users, 1))))
        self.all_item_embedding = self.embedding_item(Tensor(\
            np.reshape(np.arange(0, self.conf.num_items), (self.conf.num_items, 1))))
        self.all_user_embedding = self.squeeze(self.all_user_embedding)[:, :self.d]
        self.all_item_embedding = self.squeeze(self.all_item_embedding)[:, :self.d]
        user_input, item_input = self.squeeze(user_input), self.squeeze(item_input)
        self.user_embedding, self.item_embedding = self.embedding_user(user_input), \
            self.embedding_item(item_input)
        self.user_embedding = self.squeeze(self.user_embedding)[:, :self.d]
        self.item_embedding = self.squeeze(self.item_embedding)[:, :self.d]
        self.user_feature_matrix = self.reduce_dimension_layer(self.user_feature_matrix)
        self.item_feature_matrix = self.reduce_dimension_layer(self.item_feature_matrix)
        self.fusion_user_embedding = self.all_user_embedding + self.user_feature_matrix
        self.fusion_item_embedding = self.all_item_embedding + self.item_feature_matrix
        # ---------------layer1-----------------
        self.user_embedding_from_neighbors = get_embedding_from_neighbors(\
            self.social_neighbors_sparse_matrix2, self.fusion_user_embedding)
        self.user_embedding_from_items = get_embedding_from_neighbors(\
            self.consumed_items_sparse_matrix2, self.fusion_item_embedding)
        self.user_item_attention1 = self.exp(self.layer1_user_interest_attention(\
            self.concat((self.fusion_user_embedding, self.user_embedding_from_items))))
        self.neighbors_attention1 = self.exp(self.layer1_user_social_attention(\
            self.concat((self.fusion_user_embedding, self.user_embedding_from_neighbors))))
        self.weighted_items_attention1 = self.user_item_attention1 / (\
            self.user_item_attention1 + self.neighbors_attention1)
        self.weighted_neighbors_attention1 = self.neighbors_attention1 / (\
            self.user_item_attention1 + self.neighbors_attention1)
        self.layer1_user_embedding = 0.5 * self.fusion_user_embedding + 0.5 * (\
            self.weighted_items_attention1 * self.user_embedding_from_items + \
            self.weighted_neighbors_attention1 * self.user_embedding_from_neighbors)
        self.layer1_item_embedding = self.fusion_item_embedding
        # ----------------layer2-----------------
        self.layer2_user_embedding_from_neighbors = get_embedding_from_neighbors(\
            self.social_neighbors_sparse_matrix3, self.layer1_user_embedding)
        self.layer2_user_embedding_from_items = get_embedding_from_neighbors(\
            self.consumed_items_sparse_matrix3, self.layer1_item_embedding)
        self.items_attention2 = self.exp(self.layer2_user_interest_attention(\
            self.concat((self.layer1_user_embedding, self.layer2_user_embedding_from_items))))
        self.neighbors_attention2 = self.exp(self.layer2_user_social_attention(\
            self.concat((self.layer1_user_embedding, self.layer2_user_embedding_from_neighbors))))
        self.weighted_items_attention2 = self.items_attention2 / (\
            self.items_attention2 + self.neighbors_attention2)
        self.weighted_neighbors_attention2 = self.neighbors_attention2 / (\
            self.items_attention2 + self.neighbors_attention2)
        self.layer2_user_embedding = 0.5 * self.layer1_user_embedding + \
            0.5 * (self.weighted_items_attention2 * self.layer2_user_embedding_from_items + \
                self.weighted_neighbors_attention2 * self.layer2_user_embedding_from_neighbors)
        self.layer2_item_embedding = self.layer1_item_embedding
        self.last_user_embedding = self.concat((self.layer1_user_embedding, \
        self.layer2_user_embedding, self.all_user_embedding))
        self.last_item_embedding = self.concat((self.layer1_item_embedding, \
        self.layer2_item_embedding, self.all_item_embedding))
        self.reduced_user_embedding = self.reduce_dimension_layer1(self.last_user_embedding)
        user_input_list = (np.reshape(user_input.asnumpy(), (1, user_input.shape[0])) - 1).tolist()
        latest_user_latent = self.reduced_user_embedding[user_input_list]
        self.reduced_item_embedding = self.reduce_dimension_layer2(self.last_item_embedding)
        item_input_list = (np.reshape(item_input.asnumpy(), (1, item_input.shape[0])) - 1).tolist()
        latest_item_latent = self.reduced_item_embedding[item_input_list]
        predict_vector = self.mul(latest_user_latent, latest_item_latent)
        prediction = self.sigmoid(self.reduce_sum(predict_vector, 1))
        labels = labels.asnumpy()
        return prediction

class Myloss(_Loss):
    """Loss"""
    def __init__(self, reduction="sum"):
        super(Myloss, self).__init__(reduction)
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.square = P.Square()
        self.squeeze = P.Squeeze(axis=1)
        self.concat = P.Concat(axis=1)

    def construct(self, predict, labels):
        labels = self.squeeze(self.squeeze(labels))
        loss = self.square(predict - labels)
        sum_loss = self.reducesum(loss) * 0.5
        return self.get_loss(sum_loss)

class CustomWithLossCell(nn.Cell):
    """Custom with loss Cell"""
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, user, item, label):
        output = self._backbone(user, item, label)
        return self._loss_fn(output, label)

class ImageShowCallback(Callback):
    """Call back part"""
    def __init__(self, callback_net, net_loss):
        self.callback_net = callback_net
        self.callback_net_loss = net_loss

    def step_end(self, run_context):
        print('Loss: %.2f'%(run_context.original_args()['net_outputs']).asnumpy(), end='\t')

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    config_path = os.path.join(os.getcwd(), 'conf/yelp_Diffnetplus.ini')
    print(config_path)
    cur_conf = Parserconf(config_path)
    cur_conf.Parserconfi()
    print('Loading data...')
    #data_dict = {}
    t1 = time()
    train_data = DataModule(cur_conf, RATING_TRAIN_FILE)
    train_data.Initializerankingtrain()
    train_data_dict = train_data.Preparemodelsupplement()
    ds_rating_data, user_item_dict, cur_user_list, cur_item_list = dh.load_data(RATING_TRAIN_FILE)
    print('Time of loading data: %ds' % (time() - t1))
    net = Diffnetplus(cur_conf, train_data_dict)
    cur_loss = Myloss()
    loss_net = CustomWithLossCell(net, cur_loss)
    opt = nn.Adam(net.trainable_params(), learning_rate=0.001, weight_decay=0.0)
    model = Model(network=loss_net, optimizer=opt)
    for epoch in range(500):
        record_dict, _s_neg_dict = dh.get_train_data_from_source(user_item_dict, neg_num=8)
        test_dffnet = DatasetDiff(_s_neg_dict[USER_KEY], _s_neg_dict[ITEM_KEY], \
            _s_neg_dict[LABEL_KEY])
        ds_train = create_dataset(test_dffnet, batch_size=len(_s_neg_dict[USER_KEY]))
        # ds_train = create_dataset(test_dffnet, batch_size=1024)
        print("Epoch:%d" %(epoch + 1), end='\t')
        tt1 = time()
        print('training ', end='')
        model.train(epoch=1, train_dataset=ds_train, \
            callbacks=[ImageShowCallback(net, cur_loss)], dataset_sink_mode=False)
        out = time() - t1
        print('Time: %ds' % (out))
        if epoch % 10 == 0:
            save_checkpoint(net, ckpt_file_name='./check_point/Diffnetplus_' + str(epoch)+'.ckpt')
