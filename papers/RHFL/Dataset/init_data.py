"""
Filename: init_data.py
Author: zhangjiaming
Contact: 1692823208@qq.com
"""
import numpy as np

from Dataset.cifar import CIFAR10, CIFAR100


class Cifar10FL():
    """
    Cifar10 for FL
    """

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.1):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.construct()

    def construct(self):
        """
        Construct Participant Dataset
        """
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download,
                                self.noise_type, self.noise_rate)
        if self.train:
            data = cifar_dataobj.train_data
            target = np.array(cifar_dataobj.train_noisy_labels)
        else:
            data = cifar_dataobj.test_data
            target = np.array(cifar_dataobj.test_labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class Cifar100FL():
    """
    Cifar100 for FL
    """
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.construct()

    def construct(self):
        """
        Construct Participant Dataset
        """
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download,
                                 self.noise_type, self.noise_rate)
        if self.train:
            data = cifar_dataobj.train_data
            target = np.array(cifar_dataobj.train_noisy_labels)
        else:
            data = cifar_dataobj.test_data
            target = np.array(cifar_dataobj.test_labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset_name = 'datasets'
    dataset_root = r'./' + dataset_name
    ntype = None  # [pairflip, symmetric]
    rate = 0

    cifar100 = Cifar100FL(root=dataset_root, train=True, download=True)
    cifar10 = Cifar10FL(root=dataset_root, noise_type=ntype, noise_rate=rate, download=True)
