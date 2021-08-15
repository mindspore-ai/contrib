# coding=utf-8

import os
from .transform import TransformSegDataset
from .camvid import CamVid
from .cityscapes import Cityscapes
from mindseg.tools import dataset_dir

__all__ = ['TransformSegDataset', 'build_data_file', 'get_files_list']

_data_sets = {
    'camvidfull': (CamVid, 'CamVidFull'),
    'cityscapes': (Cityscapes, 'Cityscapes'),
}


def build_data_file(data_name: str, split: str = 'train', shard_num: int = 1, shuffle: bool = True,
                    mindrecord_path: str = None):
    assert data_name.lower() in _data_sets.keys()
    data_class, folder_name = _data_sets[data_name.lower()]
    data_set = data_class(root=os.path.join(dataset_dir(), folder_name),
                          split=split,
                          shard_num=shard_num,
                          shuffle=shuffle)
    assert mindrecord_path
    return data_set.build_data(mindrecord_path)


def get_files_list(data_name: str, split: str = 'train'):
    data_class, folder_name = _data_sets[data_name.lower()]
    data_set = data_class(root=os.path.join(dataset_dir(), folder_name), split=split)
    return data_set.images_list(), data_set.masks_list()
