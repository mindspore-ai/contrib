# Cybertron

## 介绍

Cybertron：基于MindSpore的深度分子模型通用架构

本程序由深圳湾实验室、北京大学、昌平实验室与华为MindSpore团队共同开发

开发人员：杨奕，张骏，陈迪青，张辉耀，周亚强，雷耀坤，杨立江，高毅勤等

联系方式: yangyi@szbl.ac.cn

## 软件架构

本程序为深度分子模型通用程序架构，可以支持基于图神经网络的深度分子模型。

目前本程序内置三种GNN分子模型：SchNet[1]、PhysNet[2]以及MolCT[3]。

参考文献：

[1] <https://arxiv.org/abs/1706.08566>

[2] <https://arxiv.org/abs/1902.08408>

[3] <https://arxiv.org/abs/2012.11816>

## 安装教程

本程序基于华为全场景人工智能框架MindSpore开发，使用前请先安装MindSpore：<https://mindspore.cn/>

## 使用说明

根目录下的cybertron_tutorial_CN.pdf文件为程序说明，examples目录内为使用案例：

Tutorial_00.py：数据预处理

Tutorial_01.py：基础教程（一）

Tutorial_02.py：基础教程（二）

Tutorial_03.py：归一化数据集与验证数据集的使用

Tutorial_04.py：模型参数与超参数的读取（一）

Tutorial_05.py：多任务训练（一）

Tutorial_06.py：多任务训练（二）

Tutorial_07.py：带有力的数据集的拟合

Tutorial_08.py：模型参数与超参数的读取（二）

注：Tutorial_00.py所需数据集可从以下网址下载：

dataset_qm9.npz: <http://gofile.me/6Utp7/tJ5hoDIAo>

ethanol_dft.npz: <http://gofile.me/6Utp7/hbQBofAFM>

使用Tutorial_00.py处理之后生成的数据集已在examples文件夹中，并可用于Tutorial_01.py ~ Tutorial_08.py。

## 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request

