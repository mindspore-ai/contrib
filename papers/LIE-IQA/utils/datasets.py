#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Dataset"""

import os
from PIL import Image


class LolDataset:
    """LolDataset"""
    def __init__(self, file_dir):
        self.size = 0
        self.file_dir = file_dir
        self.file_list = os.listdir(file_dir)
        self.size = len(self.file_list)

    def __getitem__(self, idx):
        lol_paths = os.path.join(self.file_dir, self.file_list[idx])
        lol_image = Image.open(lol_paths).convert('RGB')
        lol_name = self.file_list[idx]
        return (lol_image, lol_name)

    def __len__(self):
        return self.size


class LieIqaDataset:
    """LieIqaDataset"""
    def __init__(self, root_dir, txt_file):
        name_file = os.path.join(root_dir, txt_file)
        self.size = 0
        self.ref_paths = []
        self.lle_paths = []
        self.lle_names = []
        if not os.path.isfile(name_file):
            print(name_file + 'does not exist!')
        my_file = open(name_file)
        for names in my_file:
            enhanced_name = names.split(',')[0].strip()
            ref_name = names.split(',')[1].strip()
            self.lle_names.append(enhanced_name)
            enhanced_image_path = os.path.join(root_dir, 'enhanced_images', enhanced_name)
            ref_image_path = os.path.join(root_dir, 'reference_images', ref_name)
            self.ref_paths.append(ref_image_path)
            self.lle_paths.append(enhanced_image_path)
            self.size += 1
        my_file.close()

    def __getitem__(self, idx):
        ref_image = Image.open(self.ref_paths[idx]).convert('RGB')
        lle_image = Image.open(self.lle_paths[idx]).convert('RGB')
        lle_name = self.lle_names[idx]
        return (lle_image, ref_image, lle_name)

    def __len__(self):
        return self.size
