# style_transfer

## 风格迁移介绍

风格迁移（style transfer)主要是通过深度神经网络，将一副艺术风格画（style image)的风格融合到内容图像(content image)，生成一幅既保持内容图像原始内容，又保持风格图像特有风格的新图像。2015年，Gatys等人发表了文章[《A Neural Algorithm of Artistic Style》](https://arxiv.org/abs/1508.06576)，首次使用深度学习进行艺术画风格学习。
在本案例中，将了解到风格迁移的基本步骤，以及如何使用MindSpore去实现它。

## ResNet网络介绍

ResNet在2015年由微软实验室提出，斩获了当年ImageNet竞赛中分类任务第一名，目标检测第一名，获得COCO数据集中目标检测第一名，图像分割第一名。出自论文[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)。

ResNet网络提出了残差网络结构(Residual Network)来减轻退化问题，使用ResNet网络可以实现搭建较深的网络结构（突破1000层）。
ResNet网络层结构如下图所示，以输入彩色图像$224\times224$为例，首先通过数量64，卷积核大小为$7\times7$，stride为2的卷积层conv1，该层输出图片大小为$112\times112$，输出channel为64；然后通过一个$3\times3$的最大下采样池化层，该层输出图片大小为$56\times56$，输出channel为64；再堆叠4个残差网络块（conv2_x、conv3_x、conv4_x和conv5_x），此时输出图片大小为$7\times7$，输出channel为2048；最后通过一个平均池化层、全连接层和softmax，得到分类概率。

![resnet-layer](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/resnet_2.png)

对于每个残差网络块，以ResNet50网络中的conv2_x为例，其由3个Bottleneck结构堆叠而成，每个Bottleneck输入的channel为64，输出channel为256。

## 项目介绍

本项目将会创建一个基于resNet18的网络模型，利用创建的模型进行图片的风格迁移，分为如下步骤：数据读取，网络模型定义，损失函数定义，训练流程定义，执行训练以及可视化结果。
我们只使用ResNet18的backbone作为特征提取器，并未使用其头部。在代码中会首先加载mindspore提供的pretrained ResNet18模型，然后只采用ResNet18的backbone和neck组成自己的模型，并把这些组件的参数属性`requires_grad`设置成False，不会更新backbone的参数。

项目开发环境

> 系统：Ubuntu18.04.1
> GPU：NVIDIA GeForce RTX 3080

## 安装教程

1.获取仓库

```shell
 git clone https://gitee.com/shelterx/style_transfer.git
```

2.Requirements

```shell
conda create -n MindSpore python=3.7.5
conda activate MindSpore
conda install mindspore-gpu=1.7.0 cudatoolkit=11.1 -c mindspore -c conda-forge
pip install mindvision
```

## 使用说明

```shell
python style_transfer.py
```

## ISSUES

1.由于本项目需要获取resnet18的一些中间层feature map，那如何获取backbone网络的中间层模`块Cell`以及参数？

`mindspore.nn.Cell`提供了一些API去访问网络的子Cell以及参数。

1.1 访问cell

- `cells_and_names(cells=None, name_prefix="")`
    递归地获取当前Cell及输入 cells 的所有子Cell的迭代器，包括Cell的名称及其本身。注意：该方法只能递归地获取当前Cell的子cell。
- `cells()`
    返回当前Cell的子Cell的迭代器。这个方法不会递归，只会访问直接子cell。返回：Iteration类型，Cell的子Cell。
- `name_cells()`
    递归地获取一个Cell中所有子Cell的迭代器。包括Cell名称和Cell本身。
    返回：Dict[String, Cell]，Cell中的所有子Cell及其名称

1.2 访问parameters

- `mindspore.nn.Cell(expand=True)`
   返回Cell中parameter的迭代器。注意：expand (bool) – 如果为True，则递归地获取当前Cell和所有子Cell的parameter。否则，只生成当前Cell的子Cell的parameter。默认值：True。
- `parameters_and_names(name_prefix='', expand=True)`
    返回Cell中parameter的迭代器,包含参数名称和参数本身。默认递归地获取当前Cell和所有子Cell的参数及名称。
- `parameters_dict(recurse=True)`
    获取此Cell的parameter字典。默认递归地包含所有子Cell的parameter。返回OrderedDict类型，参数字典。
- `get_parameters(expand=True)`
    返回Cell中parameter的迭代器。`expand (bool)` – 如果为True，则递归地获取当前Cell和所有子Cell的parameter。否则，只生成当前Cell的子Cell的parameter。默认值：True。返回：Iteration类型，Cell的parameter。

1.3 总结

  访问子cell以及parameter的方法很多，可以分为以下几类：

1.3.1 cell

- 递归且同时返回cell及其name：`cells_and_names()`, `name_cells()`
- 非递归只返回cell本身：`cells()`

1.3.2 parameter

  都是可以选择递归或者不递归，分为两种

- 只返回parameter: `parameters_dict()`, `get_parameters()`
- 同时返回parameter和name: `parameters_and_names()`

2.本项目使用parameter不断的进行训练来生成最后的风格化图像，由于backbone参数已经冻结，parameter该放在什么地方可以让`nn.TrainOneStepCell`计算梯度实现backward？

`Parameter` 是 `Tensor` 的子类，当它们被绑定为`nn.Cell`的属性时，会自动添加到其参数列表中，并且可以通过Cell的某些方法获取，例如 `cell.get_parameters()` 。
所以只能放在网络里面，因为放在`nn.Cell`的`Parameter`对象是无法进行梯度更新的，会报`outermost network`错误。所以我们创建了一个辅助网络类，用来封装要训练的参数以及
backbone。

3.如何自定义获取中间网络输出结果？

我们通过`mindspore.nn.CellList`来构造`nn.Cell`的列表，`CellList`可以像普通Python列表一样使用，其包含的Cell均已初始化。通过遍历`CellList`可以拿到里面每一个`Cell`的输出结果。

## Citation

> @article{2016Deep,
> title={Deep Residual Learning for Image Recognition},
> author={ He, K.  and  Zhang, X.  and  Ren, S.  and  Sun, J. },
> journal={IEEE},
> year={2016}
> }

## Acknowledgement

Acknowledgement to all contributors in [MindSpore](https://gitee.com/mindspore), please check [https://www.mindspore.cn/](https://www.mindspore.cn/) and [https://gitee.com/mindspore](https://gitee.com/mindspore), welcome to star!
