# Copyright 2022 Huawei Technologies Co., Ltd
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
'''alexnet_feature_extraction'''
#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import random
import numpy as np
import cv2 as cv
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from alexnet.src.alexnet import AlexNet
from alexnet.src.config import alexnet_cifar10_cfg

#resize
def resize_image(image_dir, output_path, height, width):   # image_dir 批量处理图像文件夹 size 裁剪后的尺寸
    """
    resize images
    """
    # 获取图片路径列表
    file_path_list = []
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        file_path_list.append(file_path)
        print(file_path)
    # 逐张读取图片缩放
    dim = (width, height)
    for counter, image_path in enumerate(file_path_list):
        image = cv.imread(image_path)
        resized_img = cv.resize(image, dim)
        cv.imwrite(output_path + "img_" + str(counter) + ".png", resized_img)

image_dir1 = "./images_0612/animal"
output_path1 = "./resized_images_0612/animal/"
image_dir2 = "./images_0612/object"
output_path2 = "./resized_images_0612/object/"
size = 227
resize_image(image_dir1, output_path1, size, size)
resize_image(image_dir2, output_path2, size, size)

image_dir3 = "./test50"
output_path3 = "./resized50/"
size = 227
resize_image(image_dir3, output_path3, size, size)

#convert
def convert_to_tensor(image_di):
    """
    convert to tensors
    """
    # 获取图片路径列表
    file_path_list = []
    for filename in os.listdir(image_di):
        file_path = os.path.join(image_di, filename)
        file_path_list.append(file_path)

    img_tensor_list = []
    for image_path in enumerate(file_path_list):
        image = cv.imread(image_path)
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]#shape(1, 3, 227, 227)
        img_tensor = Tensor(image, ms.float32)
        img_tensor_list.append(img_tensor)
    return img_tensor_list


img_tensor_list1 = convert_to_tensor(output_path1)
img_tensor_list2 = convert_to_tensor(output_path2)

img_tensor_list3 = convert_to_tensor(output_path3)


#加载预训练的Alexnet模型
cfg = alexnet_cifar10_cfg
param_dict = load_checkpoint("./checkpoint_alexnet-30_3125-modified.ckpt")
network = AlexNet(cfg.num_classes, phase='test')
load_param_into_net(network, param_dict)

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
network.set_train(False)
model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})

def predict_ave_feature(tensor_list1, layer_name1):
    feature_listx = []
    for tensor in tensor_list1:
        pred_feature = model.extract_feature(tensor, layer_name1)
        #print(pred_feature.shape)
        feature_listx.append(pred_feature)
    ave_feature = sum(feature_list1)/len(feature_list1)
    return ave_feature


def predict_feature(tensor_list2, layer_name2):
    feature_listx = []
    for tensor in tensor_list2:
        pred_feature = model.extract_feature(tensor, layer_name2)
        feature_listx.append(pred_feature)
    return feature_listx


predicted = predict_feature(img_tensor_list3, 'c1')

predicted1 = predict_ave_feature(img_tensor_list1, 'c1')

def compute_size(tupleshape):
    temp = 1
    for j in range(len(tupleshape)):
        temp = temp * tupleshape[j]
    return temp


temp_shape = predicted[0].shape
flatted = compute_size(temp_shape)
temp_array = predicted[0].asnumpy().reshape(1, flatted)
sample_num = 100
sample_list = [m for m in range(flatted)]
sample_list = random.sample(sample_list, sample_num)
data = temp_array[:, sample_list]

temp_shape = predicted1[0].shape
flatted = compute_size(temp_shape)
temp_array = predicted1[0].asnumpy().reshape(1, flatted)
sample_num = 100
sample_list = [k for k in range(flatted)]
sample_list = random.sample(sample_list, sample_num)
data = temp_array[:, sample_list]


predicted2 = predict_ave_feature(img_tensor_list1, 'fc1')


layer_name = ['c1', 'c2', 'c3', 'c4', 'c5']
feature_list1 = []
feature_list2 = []
for layer in layer_name:
    predicted1 = predict_ave_feature(img_tensor_list1, layer)
    predicted2 = predict_ave_feature(img_tensor_list2, layer)
    temp_shape = predicted1[0].shape
    flatted = compute_size(temp_shape)
    temp_array1 = predicted1[0].asnumpy().reshape(1, flatted)
    temp_array2 = predicted2[0].asnumpy().reshape(1, flatted)
    sample_num = 100
    sample_list = [z for z in range(flatted)]
    sample_list = random.sample(sample_list, sample_num)
    data1 = temp_array1[:, sample_list]
    data2 = temp_array2[:, sample_list]
    feature_list1.append(data1)
    feature_list2.append(data2)
    #feature_list1.append(predict_ave_feature(img_tensor_list1,layer).mean(axis=(2,3)).asnumpy())
    #feature_list2.append(predict_ave_feature(img_tensor_list2,layer).mean(axis=(2,3)).asnumpy())

layer_name = ['c1', 'c2', 'c3', 'c4', 'c5']

feature_list = []

for layer in layer_name:
    predicted = predict_feature(img_tensor_list3, layer)
    temp_shape = predicted[0].shape
    flatted = compute_size(temp_shape)
    for i in range(len(predicted)):
        temp_array = predicted[i].asnumpy().reshape(1, flatted)
        sample_num = 100
        sample_list = [i for i in range(flatted)]
        sample_list = random.sample(sample_list, sample_num)
        data = temp_array[:, sample_list]
        feature_list.append(data)
    #feature_list1.append(predict_ave_feature(img_tensor_list1,layer).mean(axis=(2,3)).asnumpy())
    #feature_list2.append(predict_ave_feature(img_tensor_list2,layer).mean(axis=(2,3)).asnumpy())


sample_num = 100
sample_list = [n for n in range(4096)]
sample_list = random.sample(sample_list, sample_num)

fc1_feature = predict_ave_feature(img_tensor_list1, 'fc1').asnumpy()
fc2_feature = predict_ave_feature(img_tensor_list1, 'fc2').asnumpy()

fc1_feature = fc1_feature[:, sample_list]
fc2_feature = fc2_feature[:, sample_list]
feature_list1.append(fc1_feature)
feature_list1.append(fc2_feature)

fc1_feature = predict_ave_feature(img_tensor_list2, 'fc1').asnumpy()
fc2_feature = predict_ave_feature(img_tensor_list2, 'fc2').asnumpy()

fc1_feature = fc1_feature[:, sample_list]
fc2_feature = fc2_feature[:, sample_list]

feature_list2.append(fc1_feature)
feature_list2.append(fc2_feature)


sample_num = 100
sample_list = [s for s in range(4096)]
sample_list = random.sample(sample_list, sample_num)

fc1_feature_list = predict_feature(img_tensor_list3, 'fc1')
fc2_feature_list = predict_feature(img_tensor_list3, 'fc2')

for r in range(50):
    fc1_feature = fc1_feature_list[r].asnumpy()
    fc1_feature = fc1_feature[:, sample_list]
    feature_list.append(fc1_feature)

for t in range(50):
    fc2_feature = fc2_feature_list[t].asnumpy()
    fc2_feature = fc2_feature[:, sample_list]
    feature_list.append(fc2_feature)


def save_variable(v, filename):
    f = open(filename, 'wb+')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    h = pickle.load(f)
    f.close()
    return h

save_path = './test50_feature/'
savepath = save_variable(feature_list, os.path.join(save_path, 'test50_feature.pickle'))
