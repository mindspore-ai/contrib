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
'''feature_predict_abm'''
#!/usr/bin/env python
# coding: utf-8
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_variable(v, filename):
    f = open(filename, 'wb+')
    pickle.dump(v, f)
    f.close()
    return filename

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


lr_path = './lrdata_0801/'
feature_path = './features_extracted_0616/'
fea_ani = load_variavle(os.path.join(feature_path, 'features_animal.pickle'))
print(fea_ani[1].shape)
fea_obj = load_variavle(os.path.join(feature_path, 'features_object.pickle'))
#layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5','fc6','fc7']
layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7']
abm_ani = []
abm_obj = []
for i, layer in enumerate(layers):
    loadlr = load_variavle(os.path.join(lr_path, 'lrmodel_{0}.pickle'.format(layer)))
    fea_ani_nor = (fea_ani[i] - np.min(fea_ani[i])) / (np.max(fea_ani[i]) - np.min(fea_ani[i]))
    fea_obj_nor = (fea_obj[i] - np.min(fea_obj[i])) / (np.max(fea_obj[i]) - np.min(fea_obj[i]))
    abm_ani.append(list(loadlr.predict(fea_ani_nor))[0])
    abm_obj.append(list(loadlr.predict(fea_obj_nor))[0])



lr_path = './lrdata_0801/'
feature_path = './test50_feature/'
fea = load_variavle(os.path.join(feature_path, 'test50_feature.pickle'))
#layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5','fc6','fc7']
layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7']
abm = []
for i, layer in enumerate(layers):
    loadlr = load_variavle(os.path.join(lr_path, 'lrmodel_{0}.pickle'.format(layer)))
    for j in range(50):
        fea_nor = (fea[i*50 + j] - np.min(fea[i*50 + j])) / (np.max(fea[i*50 + j]) - np.min(fea[i*50 + j]))#归一化
        abm.append(list(loadlr.predict(fea_nor))[0])



abm_arr = np.array(abm).reshape(7, 50)



fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
a = zip(['red', 'orange', 'gold', 'lime', 'turquoise', 'lightskyblue', 'violet'], [1, 2, 3, 4, 5, 6, 7])

i = 0

for c, z in a:
    xs = np.arange(50)  # [0,20)之间的自然数,共20个
    ys = abm_arr[i]  # 生成20个[0,1]之间的随机数
    cs = [c] * len(xs)  # 生成颜色列表
    ax.bar(xs, ys, z, zdir='x', color=cs, alpha=0.9)  # 以zdir='x'，指定z的方向为x轴，那么x轴取值为[30,20,10,0],alpha为透明度
#   ax.bar(xs, ys, z, zdir='y', color=cs, alpha=0.8)
#   ax.bar(xs, ys, z, zdir='z', color=cs, alpha=0.8)
    i = i + 1

ax.set_xlabel('Layer')
ax.set_ylabel('Image Number')
ax.set_zlabel('abm Value')
#ax.set_title('abm Values Predicted Directly by Test Image Features', size=16)



# 虚拟数据
x = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
y = abm_arr.mean(axis=1)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(x=x, height=y, color='orange')
plt.xlabel('CNN Layer')
plt.ylabel('abm Value')
#ax.set_title("Average abm Values for Each CNN Layer", fontsize=13)



#可视化
x = list(range(len(abm_ani)))
total_width, n = 0.8, 2
width = total_width / n
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']

fig = plt.figure(figsize=(7, 5))

plt.bar(x, abm_ani, width=width, label='animal')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, abm_obj, width=width, label='object', tick_label=layers)
plt.xlabel('CNN Layer')
plt.ylabel('abm Value')
plt.legend()
plt.show()



subjects = ['Subject1', 'Subject3', 'Subject4', 'Subject5']

roi_labels = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC']#subject2的'PPA'的CNN7和CNN8疑似损坏

features = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7']


data = pd.read_pickle('./GOD/analysis_FeaturePrediction.py-' + 'Subject2' + '-' + 'V1' + '-' + 'cnn1' + '.pkl')

a = data[data.notnull()]["predicted_feature"][0][23]
a = (a - np.min(a)) / (np.max(a) - np.min(a))


loadlr = load_variavle(os.path.join(lr_path, 'lrmodel_{0}.pickle'.format('cnn1')))
temp = loadlr.predict(a.reshape(1, 100))


data = pd.read_pickle('./GOD/analysis_FeaturePrediction.py-' + 'Subject1' + '-'+'V3' + '-' + 'cnn1' + '.pkl')


abm_result = np.zeros((4, 10, 7, 50))
pre = np.zeros((50, 100))
for i, sub in enumerate(subjects):
    for j, roi in enumerate(roi_labels):
        for k, feature in enumerate(features):
            #print(sub,roi,feature)
            data = pd.read_pickle('./GOD/analysis_FeaturePrediction.py-' + sub + '-' + roi + '-' + feature + '.pkl')
            predicted = data[data.notnull()]["predicted_feature_averaged"][0]
            for z in range(50):
                loadlr = load_variavle(os.path.join(lr_path, 'lrmodel_{0}.pickle'.format(feature)))
                pre_nor = (predicted[z]- np.min(predicted[z])) / (np.max(predicted[z]) - np.min(predicted[z]))
                abm_result[i][j][k][z] = loadlr.predict(pre_nor.reshape(1, 100))

abm_ave = abm_result.mean(axis=0)



fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")
a = zip(['red', 'orange', 'gold', 'lime', 'turquoise', 'lightskyblue', 'violet'], [1, 2, 3, 4, 5, 6, 7])

i = 0

for c, z in a:
    xs = np.arange(50)  # [0,20)之间的自然数,共20个
    ys = abm_ave[0][i]  # 生成20个[0,1]之间的随机数
    cs = [c] * len(xs)  # 生成颜色列表
    ax.bar(xs, ys, z, zdir='x', color=cs, alpha=0.9)  # 以zdir='x'，指定z的方向为x轴，那么x轴取值为[30,20,10,0],alpha为透明度
#   ax.bar(xs, ys, z, zdir='y', color=cs, alpha=0.8)
#   ax.bar(xs, ys, z, zdir='z', color=cs, alpha=0.8)
    i = i + 1

ax.set_xlabel('Layer')
ax.set_ylabel('Image Number')
ax.set_zlabel('abm Value')
#ax.set_title('abm Values Predicted Directly by Test Image Features', size=16)



#计算图片特征直接预测abm与fMRI间接预测abm之间的误差
error = []
for i in range(10):
    mse = np.power((abm_ave[i]-abm_arr), 2).mean()#均方误差
    error.append(mse)



# 虚拟数据
x = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC']
y = error

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x=x, height=y, color='deepskyblue')
plt.xlabel('Brain Region')
plt.ylabel('MSE')
#ax.set_title("MSE for Each ROI", fontsize=13)
