# coding=utf-8

import os
import numpy as np
from mindspore.mindrecord import FileWriter
from .segbase import SegDataset

__all__ = ['CamVid']

seg_schema = {
    "file_name": {"type": "string"},
    "label": {"type": "bytes"},
    "data": {"type": "bytes"}
}


class CamVid(SegDataset):
    def __init__(self, root, split='train', shard_num=1, shuffle=False):
        super(CamVid, self).__init__(root, split, shard_num)
        img_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'labelsGray')

        assert split in ('train', 'val', 'test')
        if split == 'train':
            split_f = os.path.join(root, 'trainval.txt')
        elif split == 'val':
            split_f = os.path.join(root, 'test.txt')
        else:
            split_f = os.path.join(root, 'test.txt')

        self.images = []
        self.masks = []
        with open(split_f, 'r') as f:
            lines = f.readlines()
            if shuffle:
                np.random.shuffle(lines)
            for line in lines:
                img_name, mask_name = line.strip().split(' ')

                img_pth = os.path.join(img_dir, img_name)
                assert os.path.isfile(img_pth)
                self.images.append(img_pth)

                mask_pth = os.path.join(mask_dir, mask_name)
                assert os.path.isfile(mask_pth)
                self.masks.append(mask_pth)
        assert len(self.images) == len(self.masks)

    def _build_mindrecord(self, mindrecord_path):
        writer = FileWriter(file_name=mindrecord_path, shard_num=self.shard_num)
        writer.add_schema(seg_schema, "seg_schema")
        data = []
        cnt = 0
        print('number of samples:', self.num_images)
        for idx in range(len(self.images)):
            sample_ = {'file_name': os.path.basename(self.images[idx])}
            with open(self.images[idx], 'rb') as f:
                sample_['data'] = f.read()
            with open(self.masks[idx], 'rb') as f:
                sample_['label'] = f.read()
            data.append(sample_)
            cnt += 1
            if cnt % 1000 == 0:
                writer.write_raw_data(data)
                data = []
        if data:
            writer.write_raw_data(data)

        writer.commit()
        print('number of samples written:', cnt)

    def build_data(self, mindrecord_path):
        self._build_mindrecord(mindrecord_path)

    @property
    def num_images(self):
        return len(self.images)

    @property
    def num_masks(self):
        return len(self.masks)

    def images_list(self):
        return self.images

    def masks_list(self):
        return self.masks
