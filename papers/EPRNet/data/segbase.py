# coding=utf-8

import os
from abc import abstractmethod

__all__ = ['SegDataset']


class SegDataset:
    def __init__(self, root, split, shard_num=1):
        if not os.path.isdir(root):
            raise OSError(f"{root} is not a valid directory.")
        self.root = root
        self.split = split
        self.shard_num = shard_num

    @property
    def num_images(self):
        raise NotImplementedError

    @property
    def num_masks(self):
        raise NotImplementedError

    @abstractmethod
    def images_list(self):
        raise NotImplementedError

    @abstractmethod
    def masks_list(self):
        raise NotImplementedError

    @abstractmethod
    def build_data(self, path):
        raise NotImplementedError
