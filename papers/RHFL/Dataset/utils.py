"""
Filename: utils.py
Author: zhangjiaming
Contact: 1692823208@qq.com
"""
import logging
import os
import random
import sys
import numpy as np
import mindspore.dataset as data
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision import py_transforms, Border

from Dataset.init_data import Cifar10FL, Cifar100FL
from Network.Models_Def.efficientnet import efficientnet
from Network.Models_Def.resnet import ResNet10, ResNet12
from Network.Models_Def.shufflenet import ShuffleNet

seed = 0
random.seed(seed)
np.random.seed(seed)
project_path = r'/home/zhangjiaming/'


def init_logs(log_level=logging.INFO, log_path=project_path + 'Logs/', sub_name=None):
    """
    logging：https://www.cnblogs.com/CJOKER/p/8295272.html
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)  # Log等级总开关
    log_path = log_path
    mkdirs(log_path)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name is None:
        log_name = log_path + filename + '.log'
    else:
        log_name = log_path + filename + '_' + sub_name + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger


def mkdirs(dirpath):
    """
    mkdirs
    """
    os.makedirs(dirpath)



def load_cifar10_data(datadir, noise_type=None, noise_rate=0):
    """
    load cifar10 data
    """
    transform = Compose([py_transforms.ToTensor()])
    # noise_type = 'pairflip' #[pairflip, symmetric]
    # noise_rate = 0.1
    cifar10_train_ds = Cifar10FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type,
                                 noise_rate=noise_rate)
    cifar10_test_ds = Cifar10FL(datadir, train=False, download=True, transform=transform)
    x_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    x_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (x_train, y_train, x_test, y_test)


def load_cifar100_data(datadir, noise_type=None, noise_rate=0):
    """
    load cifar100 data
    """
    transform = Compose([py_transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type,
                                   noise_rate=noise_rate)
    cifar100_test_ds = Cifar100FL(datadir, train=False, download=True, transform=transform)
    x_train, y_train = cifar100_train_ds.data, cifar100_test_ds.target
    x_test, y_test = cifar100_train_ds.data, cifar100_test_ds.target
    return (x_train, y_train, x_test, y_test)


def generate_public_data_indexs(dataset, datadir, size, noise_type=None, noise_rate=0):
    """
    generate public data indexes
    """
    if dataset == 'cifar100':
        _, y_train, _, _ = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    if dataset == 'cifar10':
        _, y_train, _, _ = load_cifar10_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, noise_type=None, noise_rate=0):
    """
    get dataloader
    """
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = Cifar10FL
            normalize = py_transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = Compose([
                py_transforms.ToPIL(),
                py_transforms.Pad(padding=(4, 4, 4, 4), padding_mode=Border.REFLECT),
                py_transforms.RandomColorAdjust(brightness=noise_level),
                py_transforms.RandomCrop(32),
                py_transforms.RandomHorizontalFlip(),
                py_transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = Compose([
                py_transforms.ToTensor(),
                normalize])
        if dataset == 'cifar100':
            dl_obj = Cifar100FL
            normalize = py_transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = Compose([
                py_transforms.ToPIL(),
                py_transforms.RandomCrop(32, padding=4),
                py_transforms.RandomHorizontalFlip(),
                py_transforms.RandomRotation(15),
                py_transforms.ToTensor(),
                normalize
            ])
            transform_test = Compose([
                py_transforms.ToTensor(),
                normalize])
        # noise_type = 'pairflip'  # [pairflip, symmetric]
        # noise_rate = 0.1
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, download=False,
                          noise_type=noise_type, noise_rate=noise_rate)
        test_ds = dl_obj(datadir, train=False, download=False)

        train_dl = data.GeneratorDataset(source=train_ds, column_names=["img", "target"], shuffle=True)
        train_dl = train_dl.map(transform_train)
        train_dl = train_dl.batch(batch_size=train_bs, drop_remainder=True)

        test_dl = data.GeneratorDataset(source=test_ds, column_names=["img", "target"], shuffle=False)
        test_dl = test_dl.map(transform_test)
        test_dl = test_dl.batch(batch_size=test_bs)

    return train_dl, test_dl, train_ds, test_ds


def init_nets(n_parties, nets_name_list):
    """
    init nets
    """
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name == 'ResNet10':
            net = ResNet10()
        elif net_name == 'ResNet12':
            net = ResNet12()
        elif net_name == 'ShuffleNet':
            net = ShuffleNet()
        elif net_name == 'EfficientNet':
            net = efficientnet()
        nets_list[net_i] = net
    return nets_list


if __name__ == '__main__':
    my_logger = init_logs()
    public_data_indexs = generate_public_data_indexs(dataset='cifar10', datadir='./datasets', size=5000)
    train_l, test_l, train_s, test_s = get_dataloader(dataset='cifar10',
                                                      datadir='./datasets', train_bs=256,
                                                      noise_type='pairflip', noise_rate=0.1,
                                                      test_bs=512, dataidxs=public_data_indexs)
