"""
Filename: communication_gan.py
Author: fangxiuwen
Contact: fangxiuwen67@163.com
"""
import os
import pickle
import mindspore
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms import c_transforms
from mindspore import Tensor
import numpy as np
import scipy.io as scio

# Local adapter
def get_device_id():
    device_id = os.getenv('DEVICE_ID', '3')
    return int(device_id)

def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)

def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)

def get_job_id():
    return "Local Job"


def get_mnist_dataset():
    """
    Get MNIST dataset and divide train data and valid data
    """
    data_dir = "./Dataset/MNIST"
    data_train = create_dataset(data_dir, training=True)
    data_test = create_dataset(data_dir, training=False)
    data_train, data_validation = data_train.split([0.8, 0.2])
    return data_train, data_validation, data_test


def create_dataset(data_dir, training=True):
    """
    Create dataset
    """
    data_train = os.path.join(data_dir, 'train')
    data_test = os.path.join(data_dir, 'test')
    data_path = data_train if training else data_test
    dateset = mindspore.dataset.MnistDataset(dataset_dir=data_path, shuffle=True)
    apply_transform = c_transforms.Compose([py_vision.ToTensor(), py_vision.Normalize((0.1307,), (0.3081,))])
    dateset = dateset.map(input_columns=["image"], operations=apply_transform)
    dateset = dateset.map(input_columns=["label"], operations=c_transforms.TypeCast(mindspore.int32))
    return dateset


def pre_handle_femnist_mat():
    """
    Preprocessing EMNIST_mat
    """
    mat = scio.loadmat('Dataset/emnist-letters.mat', verify_compressed_data_integrity=False)
    #mat = sio.loadmat('Dataset/emnist-letters.mat')
    data = mat["dataset"]
    writer_ids_train = data['train'][0, 0]['writers'][0, 0]
    writer_ids_train = np.squeeze(writer_ids_train)
    x_train = data['train'][0, 0]['images'][0, 0]
    x_train = x_train.reshape((x_train.shape[0], 28, 28), order="F")
    y_train = data['train'][0, 0]['labels'][0, 0]
    y_train = np.squeeze(y_train)
    y_train -= 1
    writer_ids_test = data['test'][0, 0]['writers'][0, 0]
    writer_ids_test = np.squeeze(writer_ids_test)
    x_test = data['test'][0, 0]['images'][0, 0]
    x_test = x_test.reshape((x_test.shape[0], 28, 28), order="F")
    y_test = data['test'][0, 0]['labels'][0, 0]
    y_test = np.squeeze(y_test)
    y_test -= 1
    return x_train, y_train, writer_ids_train, x_test, y_test, writer_ids_test

def generate_partial_femnist(x, y, class_in_use=None, verbose=False):
    """
    Generate partial femnist as test set
    """
    if class_in_use is None:
        idx = np.ones_like(y, dtype=bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis=0)
    x_incomplete, y_incomplete = x[idx], y[idx]
    if verbose:
        print("Selected x shape :", x_incomplete.shape)
        print("Selected y shape :", y_incomplete.shape)
    return x_incomplete, y_incomplete

def generate_bal_private_data(x, y, n_parties=10, classes_in_use=range(11), n_samples_per_class=3, data_overlap=False):
    """
    Generate private data
    """
    priv_data = [None] * n_parties
    combined_idx = np.array([], dtype=np.int16)
    for cls in classes_in_use:
        # Get the index of eligible data
        idx = np.where(y == cls)[0]
        # Randomly pick a certain number of indices
        idx = np.random.choice(idx, n_samples_per_class * n_parties,
                               replace=data_overlap)
        # np.r_/np.c_: It is to connect two matrices by column/row, that is, add the two matrices up and down/left
        # and right, requiring the same number of columns/rows, similar to concat()/merge() in pandas.
        combined_idx = np.r_[combined_idx, idx]

        for i in range(n_parties):
            idx_tmp = idx[i * n_samples_per_class: (i + 1) * n_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp["X"] = x[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], x[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

    priv_data_save = np.array(priv_data)
    np.save('Temp/priv_data_72.npy', priv_data_save)

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = x[combined_idx]
    total_priv_data["y"] = y[combined_idx]

    with open('Temp/total_priv_data_72.pickle', 'wb') as handle:
        pickle.dump(total_priv_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return priv_data

class Femnist():
    """
    Femnist dataset class
    """
    def __init__(self, data_list, transform):
        data_x_list = data_list["X"]
        data_y_list = data_list["y"]
        imgs = []
        for index in range(len(data_x_list)):
            imgs.append((data_x_list[index], data_y_list[index]))
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        image = list(image)
        image[0] = image[0].squeeze(0)
        image = tuple(image)
        label = label.astype(np.int32)
#        <class 'tuple'> <class 'numpy.int32'>
        return image, label
    def __len__(self):
        return len(self.imgs)


class FemnistValTest():
    """
    Femnist test dataset class
    """
    def __init__(self, data_x_list, data_y_list, transform):
        imgs = []
        for index in range(len(data_x_list)):
            imgs.append((data_x_list[index], data_y_list[index]))
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        image = list(image)
        image[0] = image[0].squeeze(0)
        image = tuple(image)
        label = label.astype(np.int32)
        return image, label
    def __len__(self):
        return len(self.imgs)


class Mydata():
    """
    An abstract dataset class
    """
    def __init__(self, data_list, transform):
        data_x_list = data_list["X"]
        data_y_list = data_list["y"]
        imgs = []
        for index in range(len(data_x_list)):
            imgs.append((data_x_list[index], data_y_list[index]))
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.imgs[index]
        if self.transform is not None:
            image = self.transform(image)
        image = list(image)
        image[0] = image[0].squeeze(0)
        image = tuple(image)
        label = label.astype(np.int32)
#        <class 'tuple'> <class 'numpy.int32'>
        return Tensor(image), Tensor(label)
    def __len__(self):
        return len(self.imgs)
