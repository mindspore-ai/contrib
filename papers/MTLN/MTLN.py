"""MTLN"""
import argparse
import numpy as np
import mindspore
from mindspore import context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

parser = argparse.ArgumentParser(description='Multi-task Learning Network')
parser.add_argument('--mode', default='all', type=str,
                    help='train: train on datasests; all: train and eval datasets')
parser.add_argument('--gpu', default=0, type=int, help='GPU id')
opt = parser.parse_args()


def conv3x3(in_planes, out_planes, stride=1):
    """conv3x3"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=True)


class WideBasic(nn.Cell):
    """WideBasic"""
    def __init__(self, in_planes, planes, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=True)
        self.relu = ops.ReLU()
        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.SequentialCell([nn.Conv2d(in_planes, planes, kernel_size=1,
                                                         stride=stride, has_bias=True)])

    def construct(self, x):
        """construct"""
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Cell):
    """WideResNet, k is widen factor"""
    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        filter_net = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.conv1 = conv3x3(3, filter_net[0], stride=1)
        self.layer1 = self._wide_layer(WideBasic, filter_net[1], n, stride=2)
        self.layer2 = self._wide_layer(WideBasic, filter_net[2], n, stride=2)
        self.layer3 = self._wide_layer(WideBasic, filter_net[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter_net[3], momentum=0.9)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool_op = nn.AvgPool2d(kernel_size=9, stride=9)
        self.linear = nn.CellList([nn.SequentialCell([nn.Dense(filter_net[3], num_classes[0])])])
        # attention modules for block 1 of the task 1
        self.encoder_att = nn.CellList([nn.CellList([self.att_layer([filter_net[0], filter_net[0], filter_net[0]])])])
        self.encoder_block_extractor = nn.CellList([self.conv_layer([filter_net[0], filter_net[1]])])

        for j in range(3):
            if j < 2:       # task 2 and task 3
                self.encoder_att.append(nn.CellList([self.att_layer([filter_net[0], filter_net[0], filter_net[0]])]))
                self.linear.append(nn.SequentialCell([nn.Dense(filter_net[3], num_classes[j + 1])]))
            for ii in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter_net[ii + 1], filter_net[ii + 1],
                                                           filter_net[ii + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_extractor.append(self.conv_layer([filter_net[i + 1], filter_net[i + 2]]))
            else:
                self.encoder_block_extractor.append(self.conv_layer([filter_net[i + 1], filter_net[i + 1]]))

    def conv_layer(self, channel):
        """conv_layer"""
        conv_block = nn.SequentialCell([nn.Conv2d(in_channels=channel[0],
                                                  out_channels=channel[1],
                                                  kernel_size=3,
                                                  pad_mode='pad', padding=1),
                                        nn.BatchNorm2d(num_features=channel[1]), nn.ReLU()])
        return conv_block

    def att_layer(self, channel):
        """att_layer"""
        att_block = nn.SequentialCell([nn.Conv2d(in_channels=channel[0],
                                                 out_channels=channel[1],
                                                 kernel_size=1, padding=0),
                                       nn.BatchNorm2d(channel[1]),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=channel[1],
                                                 out_channels=channel[2],
                                                 kernel_size=1, padding=0),
                                       nn.BatchNorm2d(channel[2]),
                                       nn.Sigmoid()])
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        """_wide_layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_i in strides:
            layers.append(block(self.in_planes, planes, stride_i))
            self.in_planes = planes
        return nn.SequentialCell(*layers)

    def construct(self, x, task_k):
        """construct"""
        g_encoder = [0] * 4
        atten_encoder = [0] * 3  # 3 task
        for b in range(3):
            atten_encoder[b] = [0] * 4  # 4 block
        for i_0 in range(3):
            for jj in range(4):
                atten_encoder[i_0][jj] = [0] * 3

        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = ops.ReLU()(self.bn1(self.layer3(g_encoder[2])))

        for j in range(4):  # 4 block
            if j == 0:
                atten_encoder[task_k][j][0] = self.encoder_att[task_k][j](g_encoder[0])
                atten_encoder[task_k][j][1] = (atten_encoder[task_k][j][0]) * g_encoder[0]
                atten_encoder[task_k][j][2] = self.encoder_block_extractor[j](atten_encoder[task_k][j][1])
                atten_encoder[task_k][j][2] = self.max_pool2d(atten_encoder[task_k][j][2])
            else:
                atten_encoder[task_k][j][0] = self.encoder_att[task_k][j](
                    ops.Concat(1)((g_encoder[j], atten_encoder[task_k][j - 1][2])))
                atten_encoder[task_k][j][1] = (atten_encoder[task_k][j][0]) * g_encoder[j]
                atten_encoder[task_k][j][2] = self.encoder_block_extractor[j](atten_encoder[task_k][j][1])
                if j < 3:  #j=1 2
                    atten_encoder[task_k][j][2] = self.max_pool2d(atten_encoder[task_k][j][2])

        pred = self.avgpool_op(atten_encoder[task_k][-1][-1])
        pred = pred.view(pred.shape[0], -1)
        out = self.linear[0](pred)
        return out


class WithLossCell(nn.Cell):
    """WithLossCell"""
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=True)
        self.net = net
        self.loss_fn = loss_fn

    def construct(self, real_data, label1, task_i):
        """construct"""
        out1 = self.net(real_data, task_i)
        loss1 = self.loss_fn(out1, label1)
        return loss1

im_train_set = [0] * 3
im_test_set = [0] * 3
data_path = './dataset/'
data_name = ['Merced', 'AID', 'NWPU']
data_class = [12, 12, 12]


def create_dataset(path, name, batch_size=24, repeat_num=1, training=True, num_parallel_workers=None):
    """create_dataset"""
    data_set = ds.ImageFolderDataset(path, num_parallel_workers=num_parallel_workers, shuffle=True)
    means_aid = [0.389, 0.390, 0.392]
    stds_aid = [0.129, 0.130, 0.131]
    means_merced = [0.470, 0.470, 0.470]
    stds_merced = [0.144, 0.144, 0.144]
    means_nwpu = [0.366, 0.366, 0.366]
    stds_nwpu = [0.129, 0.127, 0.128]
    dict_mean_std = {'means_AID': means_aid, 'stds_AID': stds_aid, 'means_Merced': means_merced,
                     'stds_Merced': stds_merced, 'means_NWPU': means_nwpu, 'stds_NWPU': stds_nwpu}
    mean = dict_mean_std['means_' + name]
    std = dict_mean_std['stds_' + name]
    if training:
        trans = [vision.Decode(),
                 vision.Resize(72),
                 vision.RandomCrop(72),
                 vision.RandomHorizontalFlip(0.5),
                 vision.Rescale(1.0 / 255.0, 0.0),
                 vision.Normalize(mean=mean, std=std),
                 vision.HWC2CHW()]
    else:
        trans = [vision.Resize(72),
                 vision.Decode(),
                 vision.CenterCrop(72),
                 vision.Rescale(1.0 / 255.0, 0.0),
                 vision.Normalize(mean=mean, std=std),
                 vision.HWC2CHW()]
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)
    return data_set


for d_i in range(3):
    im_train_set[d_i] = create_dataset(data_path + data_name[d_i] + '/train', data_name[d_i], batch_size=128,
                                       repeat_num=1)
    im_test_set[d_i] = create_dataset(data_path + data_name[d_i] + '/val', data_name[d_i], batch_size=128, repeat_num=1)

# WRN model
model = WideResNet(depth=28, widen_factor=4, num_classes=data_class)

optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=0.001,
                        weight_decay=5e-5, use_nesterov=True, momentum=0.9)

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
criterion = WithLossCell(model, loss)
step = nn.TrainOneStepCell(criterion, optimizer)

total_epoch = 40
first_run = True
avg_cost = np.zeros([total_epoch, 3, 4], dtype=np.float32)

for index in range(total_epoch):
    for k in range(0, 3):
        model.set_train(True)
        cost = np.zeros(2, dtype=np.float32)
        train_dataset = im_train_set[k].create_dict_iterator(num_epochs=1)
        train_batch = im_train_set[k].get_dataset_size()
        for en, d in enumerate(train_dataset):
            train_data, train_label = d['image'], d['label']
            train_loss1 = step(train_data, train_label, k).view(-1)
            train_loss = train_loss1.mean()
            cost[0] = train_loss.asnumpy()
            test_pred1 = model(train_data, k)
            test_predict_label1 = test_pred1.argmax(axis=1)
            test_acc1 = ops.Equal()(test_predict_label1,
                                    train_label).sum().astype("float32") / float(train_data.shape[0])
            cost[1] = test_acc1.asnumpy()
            avg_cost[index][k][0] += cost[0] / float(train_batch)
            avg_cost[index][k][1] += cost[1] / float(train_batch)
        # evaluating test data
        test_dataset = im_test_set[k].create_dict_iterator(num_epochs=1)
        test_batch = im_test_set[k].get_dataset_size()
        if opt.mode == 'all':
            for d in test_dataset:
                test_data, test_label = d['image'], d['label']
                test_loss1 = step(test_data, test_label, k).view(-1)
                test_loss = test_loss1.mean()
                cost[0] = test_loss.asnumpy()

                test_pred1 = model(test_data, k)
                test_predict_label1 = test_pred1.argmax(axis=1)
                test_acc1 = ops.Equal()(test_predict_label1, test_label).sum().astype("float32") / len(test_data)
                cost[1] = test_acc1.asnumpy()
                avg_cost[index][k][2:] += cost / test_batch
        print('EPOCH: {:04d} | DATASET: {:s} || TRAIN Loss ACC: {:.4f} {:.4f}|| TEST Loss ACC: {:.4f} {:.4f}'.format(
            index, data_name[k], avg_cost[index][k][0], avg_cost[index][k][1], avg_cost[index][k][2],
            avg_cost[index][k][3]))
    if index % 5 == 0:
        mindspore.save_checkpoint(model, 'model_weights/wrn_final')
