# coding=utf-8
# adapted from:
# https://github.com/mindspore-ai/mindspore/blob/master/model_zoo/official/cv/deeplabv3/src/data/dataset.py

import cv2
import numpy as np
import mindspore.dataset as mdata

__all__ = ['TransformSegDataset']
cv2.setNumThreads(0)


class TransformSegDataset:
    def __init__(self,
                 data_file,
                 img_mean=(103.53, 116.28, 123.675),
                 img_std=(57.375, 57.120, 58.395),
                 batch_size=32,
                 crop_size=513,
                 min_scale=0.5,
                 max_scale=2.0,
                 ignore_label=255,
                 num_classes=21,
                 num_readers=2,
                 num_parallel_calls=4,
                 shard_id=None,
                 shard_num=None):
        assert max_scale > min_scale
        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.img_mean = np.array(img_mean, dtype=np.float32)
        self.img_std = np.array(img_std, dtype=np.float32)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num

    def pre_process_(self, image, label):
        # read
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # random scale
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h = int(sc * image_out.shape[0])
        new_w = int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # normalize
        image_out = (image_out - self.img_mean) / self.img_std

        # padding
        h_ = max(new_h, self.crop_size)
        w_ = max(new_w, self.crop_size)
        pad_h = h_ - new_h
        pad_w = w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                           value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]

        # left-right flip
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))  # HWC --> CHW
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    def get_transformed_dataset(self, repeat=1):
        data_set = mdata.MindDataset(dataset_file=self.data_file,
                                     columns_list=['data', 'label'],
                                     shuffle=True,
                                     num_parallel_workers=self.num_readers,
                                     num_shards=self.shard_num,
                                     shard_id=self.shard_id)
        data_set = data_set.map(operations=self.pre_process_,
                                input_columns=['data', 'label'],
                                output_columns=['data', 'label'],
                                num_parallel_workers=self.num_parallel_calls)
        data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set
