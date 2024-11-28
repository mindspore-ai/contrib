# coding=utf-8

import os
from io import BytesIO
import numpy as np
from PIL import Image
from mindspore.mindrecord import FileWriter
from .segbase import SegDataset

__all__ = ['Cityscapes']

seg_schema = {
    "file_name": {"type": "string"},
    "data": {"type": "bytes"},
    "label": {"type": "bytes"}
}


class Cityscapes(SegDataset):
    def __init__(self, root, split='train', shard_num=1, shuffle=False):
        super(Cityscapes, self).__init__(root, split, shard_num)
        self.images, self.masks = _get_city_pairs(root, split)
        assert len(self.images) == len(self.masks)
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(self.images)
            np.random.set_state(state)
            np.random.shuffle(self.masks)

        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([19, 19, 19, 19, 19, 19,
                              19, 19, 0, 1, 19, 19,
                              2, 3, 4, 19, 19, 19,
                              5, 19, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              19, 19, 16, 17, 18])  # class 19 should be ignored
        self._mapping = np.array(range(-1, len(self._key) - 1))

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

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
            white_io = BytesIO()
            mask = Image.open(self.masks[idx])
            mask = Image.fromarray(self._class_to_index(np.array(mask)).astype('uint8'))
            mask.save(white_io, 'PNG')
            mask_bytes = white_io.getvalue()
            sample_['label'] = white_io.getvalue()
            data.append(sample_)
            cnt += 1
            if cnt % 10 == 0:
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


def _get_city_pairs(folder, split='train'):
    if split in ('train', 'val', 'test'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = _get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = _get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = _get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


def _get_path_pairs(img_folder, mask_folder):
    img_paths = []
    mask_paths = []
    for root, _, files in os.walk(img_folder):
        for filename in files:
            if filename.endswith(".png"):
                imgpath = os.path.join(root, filename)
                foldername = os.path.basename(os.path.dirname(imgpath))
                maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                maskpath = os.path.join(mask_folder, foldername, maskname)
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask or image:', imgpath, maskpath)
    print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
    return img_paths, mask_paths
