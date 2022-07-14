import matplotlib.pyplot as plt
import os
import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms import c_transforms
from mindspore.dataset.transforms import py_transforms
from mindspore import nn, Tensor
import scipy.io as sio
import numpy as np
import pickle
import scipy.io as scio

"""Local adapter"""

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
    data_dir="./Dataset/MNIST"
    data_train=create_dataset(data_dir, training=True)
    data_test=create_dataset(data_dir, training=False)
    data_train, data_validation = data_train.split([0.8, 0.2])
    return data_train, data_validation, data_test


def create_dataset(data_dir, training=True):
    """
    Create dataset
    """
    data_train = os.path.join(data_dir, 'train')
    data_test = os.path.join(data_dir, 'test')
    data_path = data_train if training else data_test
    ds = mindspore.dataset.MnistDataset(dataset_dir=data_path, shuffle=True)
    apply_transform = c_transforms.Compose([py_vision.ToTensor(), py_vision.Normalize((0.1307,), (0.3081,))])
    ds = ds.map(input_columns=["image"], operations=apply_transform)
    ds = ds.map(input_columns=["label"], operations=c_transforms.TypeCast(mindspore.int32))
    return ds


def pre_handle_femnist_mat():
    """
    Preprocessing EMNIST_mat
    """
    mat = scio.loadmat('Dataset/emnist-letters.mat',verify_compressed_data_integrity=False)
    #mat = sio.loadmat('Dataset/emnist-letters.mat')
    data = mat["dataset"]
    writer_ids_train = data['train'][0, 0]['writers'][0, 0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0, 0]['images'][0, 0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order="F")
    y_train = data['train'][0, 0]['labels'][0, 0]
    y_train = np.squeeze(y_train)
    y_train -= 1
    writer_ids_test = data['test'][0, 0]['writers'][0, 0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0, 0]['images'][0, 0]
    X_test = X_test.reshape((X_test.shape[0], 28, 28), order="F")
    y_test = data['test'][0, 0]['labels'][0, 0]
    y_test = np.squeeze(y_test)
    y_test -= 1
    return X_train,y_train,writer_ids_train,X_test,y_test,writer_ids_train,writer_ids_test

def generate_partial_femnist(X, y, class_in_use = None, verbose = False):
    """
    Generate partial femnist as test set
    """
    if class_in_use is None:
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis = 0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("Selected X shape :", X_incomplete.shape)
        print("Selected y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete

def generate_bal_private_data(X, y, N_parties=10, classes_in_use=range(11),N_samples_per_class=3, data_overlap=False):
    """
    Generate private data
    """
    if False:
        priv_data = np.load('Temp/priv_data_72.npy')
        priv_data = priv_data.tolist()

        with open('Temp/total_priv_data_72.pickle', 'rb') as handle:
            total_priv_data = pickle.load(handle)
        # f = open('Src/Temp/total_priv_data.txt', 'r')
        # a = f.read()
        # total_priv_data = eval(a)
        # f.close()
    else:
        priv_data = [None] * N_parties
        combined_idx = np.array([], dtype=np.int16)
        for cls in classes_in_use:
            # Get the index of eligible data
            idx = np.where(y == cls)[0]
            # Randomly pick a certain number of indices
            idx = np.random.choice(idx, N_samples_per_class * N_parties,
                                   replace=data_overlap)
            # np.r_/np.c_: It is to connect two matrices by column/row, that is, add the two matrices up and down/left and right,
            # requiring the same number of columns/rows, similar to concat()/merge() in pandas.
            combined_idx = np.r_[combined_idx, idx]

            for i in range(N_parties):
                idx_tmp = idx[i * N_samples_per_class: (i + 1) * N_samples_per_class]
                if priv_data[i] is None:
                    tmp = {}
                    tmp["X"] = X[idx_tmp]
                    tmp["y"] = y[idx_tmp]
                    tmp["idx"] = idx_tmp
                    priv_data[i] = tmp
                else:
                    priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                    priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                    priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

        priv_data_save = np.array(priv_data)
        np.save('Temp/priv_data_72.npy', priv_data_save)

        total_priv_data = {}
        total_priv_data["idx"] = combined_idx
        total_priv_data["X"] = X[combined_idx]
        total_priv_data["y"] = y[combined_idx]

        with open('Temp/total_priv_data_72.pickle', 'wb') as handle:
            pickle.dump(total_priv_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return priv_data, total_priv_data

class Femnist():
    """
    Femnist dataset class
    """
    def __init__(self,data_list,transform):
        data_X_list = data_list["X"]
        data_Y_list = data_list["y"]
        imgs = []
        for index in range(len(data_X_list)):
            imgs.append((data_X_list[index],data_Y_list[index]))
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
        return image,label
    def __len__(self):
        return len(self.imgs)


class FemnistValTest():
    """
    Femnist test dataset class
    """
    def __init__(self,data_X_list,data_Y_list,transform):
        imgs = []
        for index in range(len(data_X_list)):
            imgs.append((data_X_list[index],data_Y_list[index]))
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
        return image,label
    def __len__(self):
        return len(self.imgs)


class Mydata():
    """
    An abstract dataset class
    """
    def __init__(self,data_list,transform):
        data_X_list = data_list["X"]
        data_Y_list = data_list["y"]
        imgs = []
        for index in range(len(data_X_list)):
            imgs.append((data_X_list[index],data_Y_list[index]))
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
        return Tensor(image),Tensor(label)
    def __len__(self):
        return len(self.imgs)
