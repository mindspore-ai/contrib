"""
Dataset for UColor
"""

import glob
import os
import random

import imageio
import mindspore.dataset as ds
import numpy as np

from utils import DataTransform


def populate_data_list(input_path, depth_path, gt_path, image_type):
    """
    populate the data list
    """
    image_list = glob.glob(os.path.join(input_path, f"*.{image_type}"))

    postfix = not os.path.exists(image_list[0].replace(input_path, depth_path))

    data_list = [
        {
            "data": image,
            "depth": image.replace(input_path, depth_path) + \
                     ("_depth_estimate.png" if postfix else ""),
            "gt": image.replace(input_path, gt_path) if gt_path else None
        } for image in image_list
    ]

    random.shuffle(data_list)
    return data_list


class DataLoader:
    """
    The dataloader
    """

    def __init__(self, input_path, depth_path, gt_path, *,
                 type_="train",
                 image_type="png",
                 patch_size=128,
                 transform=None):
        self.data_list = populate_data_list(
            input_path, depth_path, gt_path, image_type=image_type)
        self.type_ = type_
        self.patch_size = patch_size
        if transform is None:
            transform = self.transform
        self._transform = transform

    def read_image(self, uri, is_grayscale=False):
        """
        read an image and transfer it into [0,1] domain
        """
        image = imageio.imread(uri)
        image = image.transpose(2, 0, 1).astype(np.float32) if not is_grayscale else \
            image[np.newaxis].astype(np.float32)
        return image / 255.0

    def crop(self, *images):
        """
        return the crop result of an image list
        """
        h, w = images[0].shape[-2:]
        start_h = np.random.randint(0, h - self.patch_size)
        start_w = np.random.randint(0, w - self.patch_size)

        result_list = [
            image[:, start_h:start_h + self.patch_size,
                  start_w:start_w + self.patch_size].copy()
            for image in images
        ]
        return result_list

    def transform(self, *images):
        """
        data agmentation
        """
        result_list = [img for img in images]
        if random.random() > 0.5:
            result_list = [
                np.flip(img, axis=1)
                for img in result_list
            ]
        if random.random() > 0.5:
            result_list = [
                np.flip(img, axis=2)
                for img in result_list
            ]
        if random.random() > 0.5:
            result_list = [
                img.transpose((0, 2, 1))
                for img in result_list
            ]
        return result_list

    def __getitem__(self, index):
        data_dict = self.data_list[index]

        image_path = data_dict["data"]
        image = self.read_image(image_path)

        depth_path = data_dict["depth"]
        depth = self.read_image(depth_path, is_grayscale=True)

        if self.type_ == "test":
            images = np.concatenate(
                [image, image.copy(), image.copy(), depth], axis=0)
            return (images,)

        gt_path = data_dict["gt"]
        gt = self.read_image(gt_path)

        image_patch, depth_patch, gt_patch = self.crop(image, depth, gt)

        image, depth, gt = self._transform(image_patch, depth_patch, gt_patch)

        images = np.concatenate(
            [image, image.copy(), image.copy(), depth], axis=0)

        return images, gt

    def __len__(self):
        return len(self.data_list)


def make_dataset(input_path, depth_path, gt_path, *,
                 type_="train",
                 image_type="png",
                 patch_size=128,
                 transform=None,
                 batch_size=4,
                 shuffle_size=10,
                 num_parallel_workers=8):
    """
    return a mindspore dataset
    """
    dataset = DataLoader(input_path, depth_path, gt_path,
                         type_=type_,
                         image_type=image_type,
                         patch_size=patch_size,
                         transform=transform)
    if type_ == "train":
        train_dataset = ds.GeneratorDataset(
            dataset, ['input', 'gt'], num_parallel_workers=num_parallel_workers)

        train_dataset = train_dataset.map(
            operations=[DataTransform()], input_columns=['input'])

        train_dataset = train_dataset.shuffle(buffer_size=shuffle_size)
        train_dataset = train_dataset.batch(batch_size)
        return train_dataset
    if type_ == "test":
        test_dataset = ds.GeneratorDataset(
            dataset, ['input'], num_parallel_workers=num_parallel_workers)

        test_dataset = test_dataset.map(
            operations=[DataTransform()], input_columns=['input'])
        test_dataset = test_dataset.batch(batch_size)
        return test_dataset
    return None
