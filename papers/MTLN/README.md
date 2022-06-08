# MTLN-Mindspore

## 1. Introduction

Mindspore code for "Generalized Scene Classification from Small-Scale Datasets with Multi-Task Learning"

X. Zheng, T. Gong, X. Li and X. Lu, "Generalized Scene Classification From Small-Scale Datasets With Multitask
Learning," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-11, 2022.

Remote sensing images contain a wealth of spatial information. Efficient scene classification is a necessary precedent
step for further application. Despite the great practical value, the mainstream methods using deep convolutional neural networks
(CNNs) are generally pre-trained on other large datasets (such as ImageNet) thus fail to capture the specific visual characteristics of
remote sensing images. For another, it lacks generalization ability
to new tasks when training a new CNN from scratch with an existing remote sensing dataset. This paper addresses the dilemma
and uses multiple small-scale datasets to learn a generalized
model for efficient scene classification. Since the existing datasets
are heterogeneous and cannot be directly combined to train a
network, a Multi-Task Learning Network (MTLN) is developed.
The MTLN treats each small-scale dataset as an individual task,
and utilizes complementary information contained in multiple
tasks to improve the generalization. Concretely, the MTLN
consists of a shared branch for all tasks and multiple taskspecific branches with each for one task. The shared branch
extracts shared features for all tasks to achieve information
sharing among tasks. The task-specific branch distills the shared
features into task-specific features towards the optimal estimation
of each specific task. By jointly learning shared features and taskspecific features, the MTLN maintains both generalization and
discrimination ability. Two types of MTL scenarios are explored
to validate the effectiveness of the proposed method: one is to
complete multiple scene classification tasks, the other is to jointly
perform scene classification and semantic segmentation.

## 2. Start

Requirements:

```bash
Python 3.7
Mindspore
```

1. prepare your datasets and change the image path correctly
2. run "MTLN.py"

## 3. Related work

If you find the code useful in your research, please consider citing:

```bash
@ARTICLE{zheng2022generalized,  
  author={Zheng, Xiangtao and Gong, Tengfei and Li, Xiaobin and Lu, Xiaoqiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Generalized Scene Classification From Small-Scale Datasets With Multitask Learning},
  year={2022},
  volume={60},  
  number={},  
  pages={1-11}  
  }
```
