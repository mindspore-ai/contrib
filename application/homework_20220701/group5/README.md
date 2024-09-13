# 项目遇到的问题以及一些心得体会

---

## 在不同模型上的精度

| model | val_accuracy | infer_accuracy |
| :----:| :----: | :----: |
| resnet50 | 0.76 | 0.66 |
| resnet101 | 0.70 | 0.61 |
| mobilenetv2 | 0.78 | 0.78 |

---

## 遇到的问题以及解决方案

---
### 问题1：项目训练时训练精度一直非常低，没有提升

我们在一开始基于`MindSpore`官方`狗与牛角包`案例修改完代码运行时，训练时的准确度会一直低于`0.1`，我们尝试增加epoch数目，但仍然对准确度没有改善。

```python
...
--------------------
Epoch: [  47 /  50], Train Loss: [67.172], Accuracy:  0.003
epoch time: 90459.637 ms, per step time: 98.219 ms
--------------------
Epoch: [  48 /  50], Train Loss: [67.172], Accuracy:  0.003
epoch time: 90459.388 ms, per step time: 98.262 ms
--------------------
Epoch: [  50 /  50], Train Loss: [189.522], Accuracy:  0.053
epoch time: 90481.831 ms, per step time: 98.243 ms
```

之后经过排查，发现是我们对官方代码修改不完全导致的，有几处的`num_classes`忘记修改了，有时候代码可以跑起来，看起来没有问题，结果却一直异常的话，很有可能是代码隐藏着错误。

经过修改，最终实现了正常的效果。

```python
...
--------------------
Epoch: [  47 /  50], Train Loss: [67.172], Accuracy:  0.753
epoch time: 90459.637 ms, per step time: 98.219 ms
--------------------
Epoch: [  48 /  50], Train Loss: [67.172], Accuracy:  0.7693
epoch time: 90459.388 ms, per step time: 98.262 ms
--------------------
Epoch: [  50 /  50], Train Loss: [189.522], Accuracy:  0.783
epoch time: 90481.831 ms, per step time: 98.243 ms
```

---

### 问题2：部署的时候手机app界面里并没有我们数据集中的标签，即狗的种类名，比如`papillon`等。

原因是我们输出端没有正确使用图片的索引，所以我们写了一个函数来生成图片和标签对应的索引。

```python
def listdir2(path, list_name):
    i = 0
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_name[i] = file
            i += 1
```

在源代码中再应用如下代码即可解决问题，输入图片，输出狗的种类名。

```python
...
list_name2 = {}
train_path = "./datasets1/train"
listdir2(train_path, list_name2)
class_name = list_name2
plt.title(f"Predict: {class_name[result]}")
return result
...
```

## 对于模型效果提升的方案和尝试

> 如前面所示，模型最好的结果仅为80左右，因此，我们尝试了许多方案来对模型准确度进行提升。
---

### 方案1：对数据集进行预处理

通过对数据集中的图片进行识别和裁剪，使裁剪后图片的内容聚焦于识别对象本身，可以提升模型的学习效果，最终提升准确度。

对数据集调用YOLOv5接口，可以得到原数据集图片中被识别的各个对象的位置坐标。

```python
import cv2
import detect
import os
from pathlib import Path
from PIL import Image

images_path = 'D:\download\low-resolution\low-resolution'#原数据集path
path_list = os.listdir(images_path)
for i in range(len(path_list)):
    images_path1 = 'D:\download\low-resolution\low-resolution' + "\\" + path_list[i]
    detect_api = detect.DetectAPI(exist_ok=True,source=images_path1,weights='yolov5s.pt',name='exp_dog_' + i.__str__())
    label = detect_api.run()
```

再对坐标进行裁剪，即可得到裁剪后的图片。**裁剪后效果约有2%的提升，我们认为可能是因为图片输入模型后Resize方式的原因。**

---

### 方案2：使用TripletLoss损失函数，提升模型训练效果

TripletLoss 是由谷歌提出的一种损失函数，主要是用于训练差异性小的样本，比如人脸等，这与我们的数据集有相似之处，首先，各种人脸之间比较类似，五官差异比较小，同样的，各种狗之间的外观也比较相似，不同种类间差距比较小，导致模型很难分辨出不同狗的种类，为了解决这个问题，我们就可以使用TripletLoss来解决这个问题。

由于时间问题，我们仅参考网上代码实现了Triplet三元组生成算法，如下：

```python
class generate_triplets(df, num_triplets):
        def make_dictionary_for_face_class(df):
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        for _ in range(num_triplets):

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name])

        return triplets
```

若最后能成功将TripletLoss损失函数应用到模型，精度应是有提升的

---

### 方案3：使用CosFace损失函数

我们试验这个的具体原因与上面的TripletLoss类似，CosFace是腾讯AI Lab的Hao Wang等在CVPR2018.01发表，通过归一化和余弦决策边界的最大化，可实现类间差异的最大化和类内差异的最小化。

我们尝试过从Pytorch框架已有的`CosFace`实现移植到`MindSpore`，但部分`torch`上的api在`MindSpore`上找不到对应的，或者api存在差异，导致模型运行失败。部分代码如下所示：

```python
class CosFace(Cell):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(Tensor(shape=(in_features,out_features),dtype=mindspore.float32,init=One()))
        mindspore.common.initializer.XavierUniform(self.weight)

    def forward(self, input, label):
        cosine =mindspore.nn.Dense(mindspore.ops.L2Normalize(input), mindspore.ops.L2Normalize(self.weight))

        phi = cosine - self.m
        one_hot = mindspore.ops.Zeros(cosine.size())

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
```

---

## 一些尝试：

### 尝试使用ResNet101网络进行训练

我们认为效果不好的可能原因还可能是因为是微调的结果，于是尝试了使用ResNet101网络对模型进行全新训练，最终实验结果表明，模型效果不好还是因为模型结构的原因，并不是因为我们微调的原因，相关结果如下：

```python
--------------------
Epoch: [ 46 /  50], Train Loss: [1.833], Accuracy:  0.688
epoch time: 177066.042 ms, per step time: 251.514 ms
--------------------
Epoch: [ 47 /  50], Train Loss: [1.478], Accuracy:  0.693
epoch time: 178150.401 ms, per step time: 253.055 ms
--------------------
Epoch: [ 48 /  50], Train Loss: [1.524], Accuracy:  0.675
epoch time: 176963.958 ms, per step time: 251.369 ms
--------------------
Epoch: [ 49 /  50], Train Loss: [1.642], Accuracy:  0.687
epoch time: 177007.112 ms, per step time: 251.431 ms
--------------------
Epoch: [ 50 /  50], Train Loss: [1.545], Accuracy:  0.694
epoch time: 178054.869 ms, per step time: 252.919 ms
```
