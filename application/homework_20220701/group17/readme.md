# 一、模型基本情况介绍 DCGAN

## 1.1主题简介

中国山水画是中国人情思中最为厚重的沉淀。从山水画中，我们可以集中体味中国画的意境、格调、气韵和色调。再没有那一个画科能像山水画那样给国人以更多的情感。故而选择数据集时，我组成员选择山水画进行训练，希望将AI此类新兴技术与中国传统文化相结合。但在前期训练过程中，山水画生成效果不佳，故而后续又选择了图像轮廓更加清晰，图像颜色更加鲜明的simpsons数据集进行训练。
采用DCGAN网络结构，其中DCGAN网络借鉴mindspore应用实践示例中“DCGAN生成漫画头像”

## 1.2模型基本介绍

### 1.2.1数据集

1. The Simpsons数据集

   kaggle上下载

   链接：https://www.kaggle.com/datasets/ymalov/simpsons

   数量：23552张 3* 128* 128

   部分数据展示：

   ![1](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/11.png)

2. 水墨画数据集

​                kaggle上下载

​                链接：https://www.kaggle.com/datasets/myzhang1029/chinese-landscape-painting-dataset

​                数量：2192张 3* 512* 512

​                部分数据展示：

![22](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/12.png)

### 1.2.2网络

#### **生成器**

生成器将隐向量映射到数据空间。生成器通过转置卷积层将隐向量一步步转换成最后形成图像。

![3](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/3.JPG)

#### **判别器**

判别器将图像最终映射为对图片的判断，它是通过卷积层实现该功能的。判别器实际上是个CNN网络。

#### **激活函数**

激活函数将非线性的特性引入到神经网络之中。

![9](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/9.JPG)

#### Batch Normalization

生成器是DNN网络，判别器是CNN网络，它们的每一层后面都用了bn层，除了生成器模型的输出层和判别器模型的输入层，在网络其它层上都使用了Batch Normalization，使用BN可以稳定学习，有助于处理初始化不良导致的训练问题。

### 1.2.3参数调整

参数说明：

| 参数名称   | 参数含义                                   |
| ---------- | ------------------------------------------ |
| workers    | 载入数据线程数                             |
| batch_size | 批量大小                                   |
| image_size | 训练图像空间大小，所有图像都将调整为该大小 |
| nc         | 图像彩色通道数，对于彩色图像为3            |
| nz         | 隐向量的长度                               |
| ngf        | 特征图在生成器中的大小                     |
| num_epochs | 训练周期数                                 |
| lr         | 学习率                                     |
| beta1      | Adam优化器的beta1超参数                    |

其中一些参数的值没有调整，一些参数的值进行了调整

| 参数名称   | 参数值             |
| ---------- | ------------------ |
| workers    | 4                  |
| batch_size | 调整（默认128）    |
| image_size | 调整（默认64）     |
| nc         | 3                  |
| nz         | 100                |
| ngf        | 64                 |
| num_epochs | 调整（默认50）     |
| lr         | 调整（默认0.0004） |
| beta1      | 0.5                |

#### The Simpsons

1. image_size设为128，其它默认

2. 生成器学习率设为0.00004，判别器学习率设为0.0004

3. num_epochs设为300，其它默认

   这个效果最好

# 二、实现情况

## 2.1网络搭建

### 2.1.1创建网络

当处理完数据后，就可以来进行网络的搭建了。按照DCGAN论文中的描述，所有模型权重均应从`mean`为0，`sigma`为0.02的正态分布中随机初始化。

#### 生成器

生成器`G`的功能是将隐向量`z`映射到数据空间。由于数据是图像，这一过程也会创建与真实图像大小相同的 RGB 图像。在实践场景中，该功能是通过一系列`Conv2dTranspose`转置卷积层来完成的，每个层都与`BatchNorm2d`层和`ReLu`激活层配对，输出数据会经过`tanh`函数，使其返回`[-1,1]`的数据范围内。

#### 判别器

如前所述，判别器`D`是一个二分类网络模型，输出判定该图像为真实图的概率。通过一系列的`Conv2d`、`BatchNorm2d`和`LeakyReLU`层对其进行处理，最后通过`Sigmoid`激活函数得到最终概率。

DCGAN论文提到，使用卷积而不是通过池化来进行下采样是一个好方法，因为它可以让网络学习自己的池化特征。

### 2.1.2连接网络和损失函数

MindSpore将损失函数、优化器等操作都封装到了Cell中，因为GAN结构上的特殊性，其损失是判别器和生成器的多输出形式，这就导致它和一般的分类网络不同。所以我们需要自定义`WithLossCell`类，将网络和Loss连接起来。

## 2.2实例测试

### 2.2.1山水画运行情况

#### a）选择山水画作为数据集，batch_size = 128，image_size=64，epoch=10，学习率=0.0002的时候，训练效果很差。

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/a1.png)

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/a2.png)

#### b）  batch_size = 128，image_size=64，epoch=40，学习率=0.0002的时候，训练效果很差。

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/b1.png)

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/b2.png)

#### c）    batch_size = 128，image_size=64，epoch=200，学习率=0.0002的时候

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/c1.png)

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/c2.png)

#### d）    batch_size = 128，image_size=64，epoch=200，学习率=0.0008的时候

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/d.png)

 **最终发现学习率对模型效果影响较大**

### 2.2.2Simpsons运行情况

#### e）使用数据集中的一小部分，batch_size = 128，image_size = 64，epoch = 300，学习率=0.0002时，

原图像

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/e1.png)

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/e2.png)

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/e3.png)

#### f)    使用数据集中的一小部分，batch_size = 128，image_size = 128，epoch = 300,学习率=0.0002时，出现了生成器损失很大，判别器损失很小的情况。

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/f1.png)

错误原因：生成器的学习率设置为0.0004，判别器的学习率设置为0.00004

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/f2.png)

#### g)    一些失败品：

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/g1.png)

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/g2.png)

#### h)    最终，在img_size=64,batch_size=128,epoch=50时，效果较好

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/h.png)

**由于在服务器中运行，没有绘制迭代过程中G的loss值和D的loss值**

#### i)    在img_size=64,batch_size=64,epoch=70,学习率=0.0002

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/i.png)

#### j)    在img_size=64,batch_size=64,epoch=70

![img](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/j.png)

# 三、问题及解决方法

## 3.1 环境安装

安装完minicoda之后安装mindspore报错

原因：服务器已经安装过conda

解决方法（或者说是使用环境方法）：

使用conda env list显示一下环境，然后进入zhaoyu1.7环境

## 3.2 使用conda activate进入zhaoyu1.7环境出错

![图片1](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/%E5%9B%BE%E7%89%871.png)

解决方法：需要source 一下source activate 环境名称

![图片2](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/%E5%9B%BE%E7%89%872.png)

## 3. 3 编译.py文件报错

需要配置cuda的环境变量

解决方法：打开.bashrc文件

加入cuda环境变量

![图片3](https://gitee.com/etach-qs/contrib/raw/master/application/homework_hz7/group17/images/%E5%9B%BE%E7%89%873.png)

## 3.4 传文件的方法

把本地文件传到第一个服务器，然后从第一个服务器传到第二个，然后从第二个传到第三个。期间遇到很多问题，但都是些很基础的问题。