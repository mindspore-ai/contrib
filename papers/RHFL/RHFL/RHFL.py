"""
Filename: RHFL.py
Author: zhangjiaming
Contact: 1692823208@qq.com
"""
import sys
import random
from cmath import exp
from statistics import mean
from Dataset.utils import init_logs, get_dataloader, init_nets, generate_public_data_indexs, mkdirs
from loss import SCELoss, KLDivLoss
from mindspore.train.model import Model
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.nn import WithLossCell
import numpy as np


sys.path.append("..")

seed = 0
n_participants = 4  # 10
train_batch_size = 256  # 256
test_batch_size = 512
communication_epoch = 40
pariticpant_params = {
    'loss_funnction': 'SCE',
    'optimizer_name': 'Adam',
    'learning_rate': 0.001

}

"""CCR Module"""
client_confidence_reweight = True
client_confidence_reweight_loss = 'SCE'
if client_confidence_reweight:
    beta = 0.5
else:
    beta = 0

noise_type = 'symmetric'  # ['pairflip','symmetric',None]
noise_rate = 0.2
"""Heterogeneous Model Setting"""
private_nets_name_list = ['ResNet10', 'ResNet12', 'ShuffleNet', 'Mobilenetv2']
"""Homogeneous Model Setting"""
# Private_Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12']

private_dataset_name = 'cifar10'
private_data_dir = '../Dataset/datasets'
private_data_len = 10000
private_dataset_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
private_output_channel = len(private_dataset_classes)
public_dataset_name = 'cifar100'
public_dataset_dir = '../Dataset/datasets'
public_dataset_length = 5000


def evaluate_network(net, dataloader, log, loss_function):
    """

    Args:
        net:
        dataloader:
        log:
        loss_function:

    Returns:

    """
    if loss_function == 'CE':
        loss_f = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if loss_function == 'SCE':
        loss_f = SCELoss(alpha=0.1, beta=1, num_classes=10)
    model = Model(net, loss_fn=loss_f, metrics={'acc'})
    acc = model.eval(dataloader)

    log.info("Test Accuracy of the model on the test images: {} %".format(100 * acc['acc']))
    return 100 * acc['acc']


def update_model_via_private_data(net, epoch, private_dataloader, loss_function, optimizer_method,
                                  learning_rate, log):
    """

    Args:
        net:
        epoch:
        private_dataloader:
        loss_function:
        optimizer_method:
        learning_rate:
        log:

    Returns:

    """
    if loss_function == 'CE':
        loss_f = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    if loss_function == 'SCE':
        loss_f = SCELoss(alpha=0.1, beta=1, num_classes=10)

    if optimizer_method == 'Adam':
        optim = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    if optimizer_method == 'SGD':
        optim = nn.SGD(net.trainable_params(), learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4)
    t_net = nn.TrainOneStepCell(nn.WithLossCell(net, loss_f), optim)
    t_net.set_train()

    participant_local_loss_batch_list = []
    for epoch_i in range(epoch):
        for di in private_dataloader.create_dict_iterator():
            result = t_net(di['img'], di['target'])
            participant_local_loss_batch_list.append(result.asnumpy().item())
        log.info(f"Private Train Epoch: [{epoch_i} / {epoch}], "f"loss: {result}")
    return net, participant_local_loss_batch_list


if __name__ == '__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    random.seed(seed)
    np.random.seed(seed)

    # context.set_context(device_id=0)
    logger.info("Initialize Participants' Data idxs and Model")
    net_dataidx_map = {}
    for index in range(n_participants):
        idxes = np.random.permutation(50000)
        idxes = idxes[0:private_data_len]
        net_dataidx_map[index] = idxes
    logger.info(net_dataidx_map)

    net_list = init_nets(n_parties=n_participants, nets_name_list=private_nets_name_list)
    logger.info("Load Participants' Models")

    for i in range(n_participants):
        network = net_list[i]
        netname = private_nets_name_list[i]
        param_dict = mindspore.load_checkpoint('../Network/Model_Storage/' +
                                               pariticpant_params['loss_funnction'] + '/' +
                                               str(noise_type) + str(noise_rate) + '/' +
                                               netname + '_' + str(i) + '.ckpt')
        mindspore.load_param_into_net(network, param_dict)

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_indexs(dataset=public_dataset_name, datadir=public_dataset_dir,
                                                     size=public_dataset_length, noise_type=noise_type,
                                                     noise_rate=noise_rate)
    public_train_dl, _, public_train_ds, _ = get_dataloader(dataset=public_dataset_name, datadir=public_dataset_dir,
                                                            train_bs=train_batch_size, test_bs=test_batch_size,
                                                            dataidxs=public_data_indexs, noise_type=noise_type,
                                                            noise_rate=noise_rate)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    current_mean_loss_list = []  # for CCR reweight
    for epoch_index in range(communication_epoch):
        logger.info("The " + str(epoch_index) + " th Communication Epoch")

        logger.info('Evaluate Models')
        acc_epoch_list = []
        for participant_index in range(n_participants):
            netname = private_nets_name_list[participant_index]
            private_dataset_dir = private_data_dir
            _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                              train_bs=train_batch_size,
                                              test_bs=test_batch_size, dataidxs=None, noise_type=noise_type,
                                              noise_rate=noise_rate)
            network = net_list[participant_index]
            accuracy = evaluate_network(net=network, dataloader=test_dl, log=logger, loss_function='SCE')
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / n_participants

        '''
        Calculate Client Confidence with label quality and model performance
        '''
        amount_with_quality = [1 / (n_participants - 1) for i in range(n_participants)]
        weight_with_quality = []
        quality_list = []
        amount_with_quality_exp = []
        last_mean_loss_list = current_mean_loss_list
        current_mean_loss_list = []
        delta_mean_loss_list = []
        for participant_index in range(n_participants):
            network = net_list[participant_index]
            network.set_train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=private_dataset_name,
                                                                  datadir=private_data_dir,
                                                                  train_bs=train_batch_size, test_bs=test_batch_size,
                                                                  dataidxs=private_dataidx, noise_type=noise_type,
                                                                  noise_rate=noise_rate)
            if client_confidence_reweight_loss == 'CE':
                criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            if client_confidence_reweight_loss == 'SCE':
                criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=10)
            participant_loss_list = []
            for d in train_dl_local.create_dict_iterator():
                images = d['img']
                labels = d['target']
                private_linear_output = network(images)
                private_loss = criterion(private_linear_output, labels)

                participant_loss_list.append(private_loss.asnumpy().item())
            mean_participant_loss = mean(participant_loss_list)
            current_mean_loss_list.append(mean_participant_loss)
        # EXP标准化处理
        if epoch_index > 0:
            for participant_index in range(n_participants):
                delta_loss = last_mean_loss_list[participant_index] - current_mean_loss_list[participant_index]
                quality_list.append(delta_loss / current_mean_loss_list[participant_index])
            quality_sum = sum(quality_list)
            for participant_index in range(n_participants):
                amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
                amount_with_quality_exp.append(exp(amount_with_quality[participant_index]))
            amount_with_quality_sum = sum(amount_with_quality_exp)
            for participant_index in range(n_participants):
                weight_with_quality.append((amount_with_quality_exp[participant_index] / amount_with_quality_sum).real)
        else:
            weight_with_quality = [1 / (n_participants - 1) for i in range(n_participants)]



        for d in public_train_dl.create_dict_iterator():
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            '''
            Calculate Linear Output
            '''
            for participant_index in range(n_participants):
                network = net_list[participant_index]
                network.set_train()
                images = d['img']
                linear_output = network(x=images)
                softmax = ops.Softmax(axis=1)
                linear_output_softmax = softmax(linear_output)
                linear_output_target_list.append(ops.stop_gradient(linear_output_softmax.copy()))
                logsoft = ops.LogSoftmax(axis=1)
                linear_output_logsoft = logsoft(linear_output)
                linear_output_list.append(linear_output_logsoft)

            for participant_index in range(n_participants):
                network = net_list[participant_index]
                criterion = KLDivLoss()
                optimizer = nn.Adam(network.trainable_params(), learning_rate=pariticpant_params['learning_rate'])
                loss = mindspore.Tensor(0)
                train_net = nn.TrainOneStepCell(WithLossCell(network, criterion), optimizer)
                train_net.set_train()
                for i in range(n_participants):
                    if i != participant_index:
                        weight_index = weight_with_quality[i]
                        loss_batch_sample = criterion(weight_index * linear_output_list[participant_index],
                                                      weight_index * linear_output_target_list[i])
                        loss = loss + loss_batch_sample
                kl_loss_batch_list.append(loss.asnumpy().item())
            col_loss_list.append(kl_loss_batch_list)


        local_loss_batch_list = []
        for participant_index in range(n_participants):
            network = net_list[participant_index]
            network.set_train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=private_dataset_name,
                                                                  datadir=private_data_dir,
                                                                  train_bs=train_batch_size, test_bs=test_batch_size,
                                                                  dataidxs=private_dataidx, noise_type=noise_type,
                                                                  noise_rate=noise_rate)
            private_epoch = max(int(len(train_ds_local) / len(public_train_ds)), 1)

            network, private_loss_batch_list = update_model_via_private_data(net=network,
                                                                             epoch=private_epoch,
                                                                             private_dataloader=train_dl_local,
                                                                             loss_function=pariticpant_params[
                                                                                 'loss_funnction'],
                                                                             optimizer_method=pariticpant_params[
                                                                                 'optimizer_name'],
                                                                             learning_rate=pariticpant_params[
                                                                                 'learning_rate'],
                                                                             log=logger)
            mean_privat_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)
        local_loss_list.append(local_loss_batch_list)


        if epoch_index == communication_epoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(n_participants):
                _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_data_dir,
                                                  train_bs=train_batch_size,
                                                  test_bs=test_batch_size, dataidxs=None, noise_type=noise_type,
                                                  noise_rate=noise_rate)
                network = net_list[participant_index]
                accuracy = evaluate_network(net=network, dataloader=test_dl, log=logger, loss_function='SCE')
                acc_epoch_list.append(accuracy)
            acc_list.append(acc_epoch_list)
            accuracy_avg = sum(acc_epoch_list) / n_participants

        if epoch_index % 5 == 0 or epoch_index == communication_epoch - 1:
            mkdirs('./test/Performance_Analysis/' + pariticpant_params['loss_funnction'])
            mkdirs('./test/Model_Storage/' + pariticpant_params['loss_funnction'])
            mkdirs('./test/Performance_Analysis/' + pariticpant_params['loss_funnction'] + str(noise_type))
            mkdirs('./test/Model_Storage/' + pariticpant_params['loss_funnction'] + str(noise_type))
            mkdirs('./test/Performance_Analysis/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(
                noise_rate))
            mkdirs('./test/Model_Storage/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(
                noise_rate))

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save('./test/Performance_Analysis/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(
                noise_rate)
                    + '/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save('./test/Performance_Analysis/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(
                noise_rate)
                    + '/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save('./test/Performance_Analysis/' + pariticpant_params['loss_funnction'] + '/' + str(noise_type) + str(
                noise_rate)
                    + '/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(n_participants):
                netname = private_nets_name_list[participant_index]
                network = net_list[participant_index]
                mindspore.save_checkpoint(network, './test/Model_Storage/' + pariticpant_params['loss_funnction'] + '/'
                                          + str(noise_type) + str(noise_rate) + '/'+
                                          netname + '_' + str(participant_index) + '.ckpt')
