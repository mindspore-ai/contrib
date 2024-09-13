# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in run_pretrain.py
"""
import os
import random
import json
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import numpy as np
from PIL import Image
from collection import namedtuple

from src.tokenization import FullTokenizer


class MyDataset:
    """
    My dataset
    """
    def __init__(self, data_path, txt_name, train=True):
        self.tokenizer = FullTokenizer('vocab_2.txt')
        with open(os.path.join(data_path, txt_name), 'r') as f:
            self.data_dict = json.load(f)
        self.new_loader = []
        times = 0
        for i in self.data_dict.keys():
            img = Image.open(os.path.join(data_path, "train" if train else "test", i + '.jpg')) \
                .resize((512, 512), Image.ANTIALIAS)
            if np.shape(img)[-1] != 3:
                img = img.convert('RGB')
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            assert np.shape(img)[0] == 3, "3 channels image required."
            # patches_number = 16
            # assert np.abs(np.sqrt(patches_number) - int(np.sqrt(patches_number))) <= 1e-8, 'patches_number have to be sqrtable.'
            # patches_dim = int(np.sqrt(patches_number))
            # patches = []
            # wide = np.shape(img)[0] // patches_dim
            # high = np.shape(img)[1] // patches_dim
            # for h in range(patches_dim):
            #     for l in range(patches_dim):
            #         patches.append(img[h * wide:(h + 1) * wide, l * high:(l + 1) * high])
            #         pass
            #     pass
            for j in range(5):  # 5 captions
                triplet = ''
                for p in self.data_dict[i]['triplet'][j]['sng']:
                    triplet += " ".join(p)
                    triplet += "."
                self.new_loader.append([img] + [
                    self.data_dict[i]['captions'][j] + " ".join(self.data_dict[i]['obj'][j]) + "." + " ".join(
                        self.data_dict[i]['relations'][j]) + "." + triplet] + [1])
            ## 创建负样本
            if train:
                random_id = random.choice(list(self.data_dict.keys()))
                if random_id != i:
                    for m in range(5):
                        triplet = ''
                        for p in self.data_dict[random_id]['triplet'][m]['sng']:
                            triplet += " ".join(p)
                            triplet += "."
                        self.new_loader.append(
                            [img] +
                            [self.data_dict[random_id]['captions'][m] +
                             " ".join(self.data_dict[random_id]['obj'][m]) +
                             "." +
                             " ".join(self.data_dict[random_id]['relations'][m]) +
                             "." + triplet] + [0])
            times += 1
            if times == 1000:
                break
        ## EVAL
        # counts = 0
        # self.val_loader = []
        # for i in self.new_loader:
        #     for j in range(100):
        #         if counts == j:
        #             self.val_loader.append([self.new_loader[5 * j][0]] + [i[1]] + [1])
        #             pass
        #         else:
        #             self.val_loader.append([self.new_loader[5 * j][0]] + [i[1]] + [0])
        #             pass
        #         pass
        #     counts += 1
        # self.new_loader = copy.deepcopy(self.val_loader)

    def __len__(self):
        return len(self.new_loader)

    def __getitem__(self, item):
        frame = self.new_loader[item][0]
        frame_mask = np.ones((1,))
        # if len(self.new_loader[item][1]) >= 112:
        #     text = self.new_loader[item][1][:112]
        #     pass
        # else:
        #     text = self.new_loader[item][1] + " " * (112 - len(self.new_loader[item][1]))
        #     pass
        # assert len(text) == 112, "Text Length ERROR."
        token = self.tokenizer.tokenize(self.new_loader[item][1])
        text_mask = np.zeros((112,))
        text_mask[:np.shape(token)[0]] = 1
        text = token[:112]
        Data = namedtuple("data", ['input_ids', 'input_mask', 'frame', 'frame_mask', 'label_ids'])
        data = Data(
            input_ids=text,
            input_mask=text_mask,
            frame=frame,
            frame_mask=frame_mask,
            label_ids=self.new_loader[item][2]
        )
        return data


def create_classification_dataset(batch_size=1, assessment_method="accuracy",
                                  data_file_path=None, do_shuffle=True):
    """create finetune or evaluation dataset"""
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op_float = C.TypeCast(mstype.float32)
    dataset_gen = MyDataset(data_file_path, "captions/train_v2")
    dataset = ds.GeneratorDataset(
        dataset_gen,
        column_names=["input_ids", "input_mask", "frame", "frame_mask", "label_ids"],
        shuffle=do_shuffle,
        num_parallel_workers=1
    )
    if assessment_method == "Spearman_correlation":
        dataset = dataset.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        dataset = dataset.map(operations=type_cast_op, input_columns="label_ids")
    dataset = dataset.map(operations=type_cast_op, input_columns="frame_mask")
    dataset = dataset.map(operations=type_cast_op_float, input_columns='frame')
    dataset = dataset.map(operations=type_cast_op, input_columns="input_mask")
    dataset = dataset.map(operations=type_cast_op, input_columns="input_ids")
    data_set = dataset.batch(batch_size, drop_remainder=True)
    #################################################################################################
    # type_cast_op = C.TypeCast(mstype.int32)
    # data_set = ds.TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
    #                               columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"],
    #                               shuffle=do_shuffle)
    # if assessment_method == "Spearman_correlation":
    #     type_cast_op_float = C.TypeCast(mstype.float32)
    #     data_set = data_set.map(operations=type_cast_op_float, input_columns="label_ids")
    # else:
    #     data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    # data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    # data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    # data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    # # apply batch operations
    # data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set
