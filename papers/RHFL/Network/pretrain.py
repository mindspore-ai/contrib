"""
Filename: pretrain.py
Author: zhangjiaming
Contact: 1692823208@qq.com
"""
import random
import sys

import mindspore
import mindspore.nn as nn
from mindspore import Model
import numpy as np
from numpy.core.fromnumeric import mean
from mindvision.engine.callback import LossMonitor
from Dataset.utils import mkdirs, init_logs, init_nets, get_dataloader
from loss import SCELoss

sys.path.append("..")


seed = 0
n_participants = 4
train_batch_size = 256
test_batch_size = 512
pretrain_epoch = 40
private_data_len = 10000
pariticpant_params = {
    'loss_funnction': 'SCE',
    'optimizer_name': 'Adam',
    'learning_rate': 0.001
}

"""Noise Setting"""
noise_type = 'symmetric'  # ['pairflip','symmetric',None]
noise_rate = 0.2
"""Heterogeneous Model Setting"""
nets_name_list = ['ResNet10', 'ResNet12', 'ShuffleNet', 'EfficientNet']
"""Homogeneous Model Setting"""
# Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12']
dataset_name = 'cifar10'
dataset_dir = '../Dataset/datasets'
dataset_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
output_channel = len(dataset_classes)


def pretrain_network(epoch, net, data_loader, loss_function, optimizer_name, learning_rate):
    """

    Args:
        epoch:
        net:
        data_loader:
        loss_function:
        optimizer_name:
        learning_rate:

    Returns:

    """
    if loss_function == 'CE':
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if loss_function == 'SCE':
        criterion = SCELoss(alpha=0.1, beta=1, num_classes=10)

    if optimizer_name == 'Adam':
        optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    if optimizer_name == 'SGD':
        optimizer = nn.SGD(net.trainable_params(), learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4)
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={'acc'})
    model.train(epoch, data_loader, callbacks=[LossMonitor(0.001, 1)])
    return net


def evaluate_network(net, dataloader, loss_function):
    """

    Args:
        net:
        dataloader:
        loss_function:

    Returns:

    """
    if loss_function == 'CE':
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if loss_function == 'SCE':
        criterion = SCELoss(alpha=0.1, beta=1, num_classes=10)
    model = Model(net, loss_fn=criterion, metrics={'acc'})
    acc = model.eval(dataloader)

    print("Test Accuracy of the model on the test images: {} %".format(100 * acc['acc']))
    return 100 * acc['acc']


if __name__ == '__main__':
    # context.set_context(device_id=0)
    mkdirs('./Model_Storage/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(noise_rate))
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    random.seed(seed)
    np.random.seed(seed)

    logger.info("Load Participants' Data and Model")
    net_dataidx_map = {}
    for index in range(n_participants):
        idxes = np.random.permutation(50000)
        idxes = idxes[0:private_data_len]
        net_dataidx_map[index] = idxes
    logger.info(net_dataidx_map)
    net_list = init_nets(n_parties=n_participants, nets_name_list=nets_name_list)

    logger.info('Pretrain Participants Models')
    for index in range(n_participants):
        train_dl_local, test_dl, train_ds_local, test_ds = get_dataloader(dataset=dataset_name, datadir=dataset_dir,
                                                                          train_bs=train_batch_size,
                                                                          test_bs=test_batch_size,
                                                                          dataidxs=net_dataidx_map[index],
                                                                          noise_type=noise_type, noise_rate=noise_rate)
        network = net_list[index]
        netname = nets_name_list[index]
        logger.info('Pretrain the ' + str(index) + ' th Participant Model with N_training: ' + str(len(train_ds_local)))
        network = pretrain_network(epoch=pretrain_epoch, net=network, data_loader=train_dl_local,
                                   loss_function=pariticpant_params['loss_funnction'],
                                   optimizer_name=pariticpant_params['optimizer_name'],
                                   learning_rate=pariticpant_params['learning_rate'])
        logger.info('Save the ' + str(index) + ' th Participant Model')
        mindspore.save_checkpoint(network, './Model_Storage/' +
                                  pariticpant_params['loss_funnction'] + '/' + str(noise_type) +
                                  str(noise_rate) + '/' + netname + '_' + str(index) + '.ckpt')

    logger.info('Evaluate Models')
    test_accuracy_list = []
    for index in range(n_participants):
        _, test_dl, _, _ = get_dataloader(dataset=dataset_name, datadir=dataset_dir, train_bs=train_batch_size,
                                          test_bs=test_batch_size, dataidxs=net_dataidx_map[index])
        network = net_list[index]
        netname = nets_name_list[index]

        param_dict = mindspore.load_checkpoint(
            './Model_Storage/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(
                noise_rate) + '/' + netname + '_' + str(index) + '.ckpt')
        mindspore.load_param_into_net(network, param_dict)
        output = evaluate_network(net=network, dataloader=test_dl, loss_function='SCE')
        test_accuracy_list.append(output)
    print('The average Accuracy of models on the test images:' + str(mean(test_accuracy_list)))
