# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================
'''dataset.py'''

import os
import copy
import random
from collections import defaultdict

import numpy as np
import mindspore.dataset as ds

from src.utils.config import config

from .transforms import build_train_transforms, build_test_transforms
from .datasets_define import (Market1501, DukeMTMCreID, MSMT17, CUHK03)


def init_dataset(name, **kwargs):
    """Initializes an image dataset."""
    all_image_datasets = {
        'market1501': Market1501,
        'cuhk03': CUHK03,
        'dukemtmcreid': DukeMTMCreID,
        'msmt17': MSMT17,
    }
    avai_datasets = list(all_image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return all_image_datasets[name](**kwargs)


class RandomIdentitySampler(ds.Sampler):  # torch original
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[int(pid)].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if not batch_idxs_dict[pid]:
                    # len == 0
                    avai_pids.remove(pid)

        return iter(final_idxs * 2)

    def __len__(self):
        return self.length


def dataset_creator(
        root='',
        dataset=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        mode=None
):
    '''
    create and preprocess data for train and evaluate
    '''
    if dataset is None:
        raise ValueError('dataset must not be None')
    dataset_ = init_dataset(
        name=dataset,
        root=root,
        mode=mode,
        cuhk03_labeled=cuhk03_labeled,
        cuhk03_classic_split=cuhk03_classic_split,
    )

    num_pids = dataset_.num_train_pids

    if mode == 'train':
        device_num, rank_id = _get_rank_info()

        if config.sampler == 'pk':
            sampler = RandomIdentitySampler(
                dataset_, config.batch_size_train, config.num_instances)
        else:
            sampler = ds.RandomSampler()

        device_num, rank_id = _get_rank_info()
        if isinstance(device_num, int) and device_num > 1:
            data_set = ds.GeneratorDataset(
                dataset_, ['img', 'pid'],
                sampler=sampler, num_parallel_workers=workers,
                num_shards=device_num, shard_id=rank_id, shuffle=True)
        else:
            data_set = ds.GeneratorDataset(
                dataset_, ['img', 'pid'],
                sampler=sampler, num_parallel_workers=workers)

        transforms = build_train_transforms(height=height, width=width, transforms=transforms,
                                            norm_mean=norm_mean, norm_std=norm_std)
        data_set = data_set.map(operations=transforms, input_columns=['img'])
        data_set = data_set.batch(
            batch_size=batch_size_train, drop_remainder=True)
        return num_pids, data_set

    data_set = ds.GeneratorDataset(dataset_, ['img', 'pid', 'camid'],
                                   num_parallel_workers=workers)
    transforms = build_test_transforms(height=height, width=width,
                                       norm_mean=norm_mean, norm_std=norm_std)
    data_set = data_set.map(operations=transforms, input_columns=['img'])
    data_set = data_set.batch(batch_size=batch_size_test, drop_remainder=False)

    return num_pids, data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
