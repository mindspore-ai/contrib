"""
This is the dataset for training the Zero-DCE.
"""

import glob
import os
import random

from PIL import Image
import numpy as np
import mindspore.dataset as ds

import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype


def populate_train_list(lowlight_images_path, image_type):
    """
    Randomly shuffle the training dataset for training.
    """
    image_list_lowlight = glob.glob(os.path.join(lowlight_images_path, f"*.{image_type}"))

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class LowlightLoader:
    """
    The dataset for mindspore training.
    """

    def __init__(self, lowlight_images_path, _type="train", *, image_type="jpg"):
        self.train_list = populate_train_list(lowlight_images_path, image_type)
        self.size = 512 if _type == "test" else 256
        self._type = _type

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]

        data_lowlight = Image.open(data_lowlight_path)

        data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

        data_lowlight = (np.asarray(data_lowlight) / 255.0)

        data_lowlight = data_lowlight.transpose(2, 0, 1)

        if self._type == "train":
            return data_lowlight, data_lowlight.copy()
        if self._type == "test":
            return (data_lowlight,)
        return None

    def __len__(self):
        return len(self.data_list)


def make_dataset(path, batch_size=8, shuffle_size=10, *, image_type="jpg"):
    """
    Make a dataset for training.
    """
    train_dataset = ds.GeneratorDataset(
        LowlightLoader(path, image_type=image_type),
        ["data", "label"]
    )

    type_cast_op_image = C.TypeCast(mstype.float32)
    train_dataset = train_dataset.map(operations=[type_cast_op_image], input_columns="data")
    train_dataset = train_dataset.map(operations=[type_cast_op_image], input_columns="label")

    train_dataset = train_dataset.shuffle(buffer_size=shuffle_size)
    train_dataset = train_dataset.batch(batch_size)

    return train_dataset


def make_test_dataset(path, batch_size=8, *, image_type="jpg"):
    """
    Make a dataset for testing.
    """
    test_dataset = ds.GeneratorDataset(
        LowlightLoader(path, _type="test", image_type=image_type),
        ["data"]
    )

    type_cast_op_image = C.TypeCast(mstype.float32)
    test_dataset = test_dataset.map(operations=[type_cast_op_image], input_columns="data")

    test_dataset = test_dataset.batch(batch_size)

    return test_dataset
