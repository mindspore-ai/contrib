"""dataset"""
import numpy as np
import pandas as pd
from mindspore import dataset as ds
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import Callback
from mindspore.nn.loss.loss import _Loss
from mindspore import Model

#DATA_DIR
RATING_TRAIN_FILE = "../data/yelp/yelp.val.rating"
RATING_TEST_FILE = "../data/yelp/yelp.val.rating"
RATING_VAL_FILE = "./data/yelp/yelp.val.rating"
TEST = "./data/yelp/test"
LINK_TRAIN_FILE = "../data/yelp/yelp.links"

user_num = 17237
item_num = 38342

def load_data(file_path):
    """load data"""
    data = {}
    with open(file_path) as f:
        df = pd.read_csv(f, sep=' ')
        df.columns = ['user_id', 'item_id', 'label']
    data['user_id_col'] = df['user_id'].values
    data['item_id_col'] = df['item_id'].values
    data['label_id_col'] = df['label'].values
    user_list_id = sorted(df['user_id'].unique())
    item_list_id = sorted(df['item_id'].unique())

    return data, user_list_id, item_list_id

ds_rating_val, user_list, item_list = load_data(TEST)

class DatasetDiff:
    """dataset class"""
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
        user = np.reshape(user, (-1, 1))
        label = np.reshape(label, (-1, 1))
        item = np.reshape(item, (-1, 1))

        return user, item, label

    def __len__(self):
        return len(self.user_col)

test_dffnet = DatasetDiff(ds_rating_val['user_id_col'], \
    ds_rating_val['item_id_col'], ds_rating_val['label_id_col'])
samplelist = ds.RandomSampler(num_samples=None, replacement=False)

#def create_dataset(data_source, batch_size=2):
def create_dataset(data_source):
    """create dataset"""
    input_data = ds.GeneratorDataset(data_source, \
        column_names=['user_id_col', 'item_id_col', 'label_id_col'], sampler=samplelist)
    input_data = input_data.repeat(1)
    return input_data

####  MODEL PART ####
class DiffNetplus(nn.Cell):
    """diffnet data"""
    def __init__(self, num_users, num_items, num_factors, mf_dim):
        super(DiffNetplus, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.mf_dim = mf_dim
        # Initializer for embedding layers
        self.embedding_initializer = "normal"

        self.embedding_user = nn.Embedding(self.num_users, self.num_factors, \
            embedding_table=self.embedding_initializer)
        self.embedding_item = nn.Embedding(self.num_items, self.num_factors, \
            embedding_table=self.embedding_initializer)
        # ops definition
        self.mul = P.Mul()
        self.squeeze = P.Squeeze(axis=1)
        self.concat = P.Concat(axis=1)

    def construct(self, user_input, item_input):
        """construct part"""
        embedding_user = self.embedding_user(user_input)
        embedding_item = self.embedding_item(item_input)
        mf_user_latent = self.squeeze(embedding_user)[:, :self.num_factors]
        mf_item_latent = self.squeeze(embedding_item)[:, :self.num_factors]

        # Element-wise multiply
        mf_vector = self.mul(mf_user_latent, mf_item_latent)
        predict_vector = mf_vector
        logits = self.logits_dense(predict_vector)
        # Print model topology.
        return logits


class Myloss(_Loss):
    """Myloss"""
    def __init__(self, network):
        super(Myloss, self).__init__()
        self.network = network
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.squeeze = P.Squeeze(axis=1)
        self.concat = P.Concat(axis=1)

    def construct(self, batch_users, batch_items, labels):
        """construct part"""
        predict = self.network(batch_users, batch_items)
        #predict = self.concat((self.zeroslike(predict), predict))
        labels = self.squeeze(labels)
        #loss = self.loss(predict, labels)
        self.loss = self.square(predict-labels)*0.5
        #mean_loss = self.mul(self.reducesum(loss), \
        # self.reciprocal(self.reducesum(valid_pt_mask)))
        mean_loss = self.mul(self.reducesum(loss), self.reciprocal(loss))
        sum_loss = self.reducesum(mean_loss)
        return sum_loss

class CustomWithLossCell(nn.Cell):
    """Custom with loss cell"""
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, user, item, label):
        """construct part"""
        #output = self._backbone(user, item)   # unused output (Junwei)
        return self._loss_fn(user, item, label)

#loss_net = Myloss(net)
#opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
#model = Model(net, loss_net, opt)

class ImageShowCallback(Callback):
    """Call back function"""
    def __init__(self, callback_net, net_loss):
        self.net = callback_net
        self.net_loss = net_loss

    def step_end(self):
        #a=1
        print(self.net_loss)

epoch = 10
#imageshow_cb = ImageShowCallback(net,loss_net)
#model.train(epoch, ds_test, callbacks=[imageshow_cb], dataset_sink_mode=False)

net = DiffNetplus(user_num, item_num, 16, 16)
loss = Myloss(net)
# build loss network
loss_net = CustomWithLossCell(net, loss)
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
model = Model(network=loss_net, optimizer=opt)
#ds_train = create_multilabel_dataset(num_data=160)
ds_train = create_dataset(test_dffnet)
model.train(epoch=1, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
