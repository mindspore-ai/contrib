#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import re
import os
import numpy as np
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
from util.misc import norm2


class Icdar15Text(TextDataset):

    def __init__(self, data_root, k, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.k = k

        self.image_root = os.path.join(data_root, 'imgs', 'training' if is_training else 'test')
        self.annotation_root = os.path.join(data_root, 'annotations', 'training' if is_training else 'test')
        self.image_list = os.listdir(self.image_root)

        p = re.compile('.rar|.txt')
        self.image_list = [x for x in self.image_list if not p.findall(x)]
        p = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        self.annotation_list = ["gt_"+'{}'.format(p.sub("", img_name)) for img_name in self.image_list]

    @staticmethod
    def parse_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path+".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, gt[:8]))
            xx = [x1, x2, x3, x4]
            yy = [y1, y2, y3, y4]

            label = gt[-1].strip().replace("###", "#")
            pts = np.stack([xx, yy]).T.astype(np.int32)

            d1 = norm2(pts[0] - pts[1])
            d2 = norm2(pts[1] - pts[2])
            d3 = norm2(pts[2] - pts[3])
            d4 = norm2(pts[3] - pts[0])
            if min([d1, d2, d3, d4]) < 2:
                continue
            polygons.append(TextInstance(pts, 'c', label))

        return polygons

    def __getitem__(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        try:
            # Read annotation
            annotation_id = self.annotation_list[item]
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_txt(annotation_path)
        except Exception as e:
            print(e)
            polygons = None

        return self.get_training_data(image, polygons, self.k, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

