# <center>深度学习与肺部X光图像分类</center>
## 背景介绍  
•当下人工智能和各行各业都有了很大程度的交叉，以人工智能相关技术为基础实现的针对不同领域专业性的产品、设备应运而生，因此本次小组选题也沿着这一思路，选取到了智慧医疗这一层面。  
## 应用前景
• 本模型可以用于肺炎x光片的识别，患者或者医生可以使用搭载本模型的设备来简单识别一张肺部x光片，从而初步检查是否患有肺炎。  
• 在此模型的基础上，可以进行拓展。由于不同肺炎的病灶不同，以及肺炎的严重程度也会反映在x光片中，当我们加强模型的能力之后，模型可以分辨出不同的肺炎类型，并针对该患者的肺炎，为患者推荐治疗该种肺炎更厉害的专家供患者选择。
## AlexNet介绍
### 原文地址
https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
### 基本概念  
• AlexNet网络包括了6000万个参数和65000万个神经元，5个卷积层，在一些卷积层后面还有池化层，3个全连接层，输出为 的softmax层  
• 激活函数使用ReLU激活函数  

![ReLU.png](https://s2.loli.net/2022/07/03/wbPyE3AkIaStl5M.png)  
 
• 重叠池化  
通常情况下，在卷积神经网络中我们设置池化层参数为步幅s=2,窗口大小f=2，这相当于对图像进行下采样，而AlexNet使用了重叠的池化层，设置s=2,f=3。在这以前大部分是使用平均池化，到了Alexnet中就全部使用最大池化（max pool)，这可以很好的解决平均池化的模糊问题。同时Alexnet提出让卷积核扫描步长比池化核的尺寸小，这样在池化层的输出之间会有重叠和覆盖，在一定程度上提升了特征的丰富性。   
• Dropout  
Dropout是以一定的概率使神经元的输出为0，AlexNet设置概率为0.5，AlexNet在第1,2个全连接网络中使用了dropout,这使得迭代收敛的速度增加了一倍  
• 归一化  
AlexNet在激活函数之外又使用了局部归一化，公式如下：  
![公式.png](https://s2.loli.net/2022/07/03/rSa4YKfMRET56lw.png)  
  其中aix,y表示第i个卷积在(x,y)产生的值然后应用ReLU激活函数的结果，n表示相邻的几个卷积核，N表示这一层总的卷积核数量。k,n,α,β都是超参数，k=2,n=5,α=1e-4,β=0.75.这种归一化操作实现了某种形式的横向抑制，这也是真实神经元的某种行为启发。这种具有对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得更大，而对响应比较小的值更加加以抑制，从而增强模型的泛化能力，这和让更加明显的特征更加明显，很细微不明显的特征加以抑制是一个道理。
### 网络结构（有调整）
• 原网络使用两个GPU，本次更改为一个GPU
![网络顺序.png](https://s2.loli.net/2022/07/03/Q73eF56RXGqn2ZO.png)  

• 更详细的结构  
![流程.png](https://s2.loli.net/2022/07/03/hY37QTjXwZCr6Bp.png)  

##  数据集选取
• 数据集从开源网站kaggle中选取，选取广州市某医院肺部x光片（针对肺炎），分为未患病（清晰肺部）和患肺炎（肺部x光片较为模糊）两类  
• 由数据集的特征可知，这是一个二分类问题  
• 众所周知，x光片是一种黑白图片（笼统意义上的），因此在进行归一化处理时有其特殊之处（在后文开发思路中具体介绍）  
• 数据集链接：https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  
## 开发过程
### 选题思路
• 在进行开发之前接触到了MindSpore框架下实现的ViT模型图像分类，在学习了该案例之后，小组决定选用AlexNet网络模型替换ViT模型，来实现其他图片数据集的分类问题  
• AlexNet网络结构简单，理解起来更容易，贴合小组情况
### 数据集的处理
• 注意到MindSpore框架提供了mindrecord数据类型，且这种数据类型能够让MindSpore框架下实现的模型更好地训练  
• 读取解析MindRecord数据文件构建数据集(只展示如何将图像转为MindRecord数据以及如何读取）（具体见下文“如何使用MindSpore框架未定义好的其他数据集”）  
• 图片转mindrecord（ test.py)
 ```python
import glob
import os.path

from mindspore.mindrecord import FileWriter

data_record_path = r"convert_dataset_to_mindrecord/data_to_mindrecord/reasoning/test.mindrecord"
writer = FileWriter(file_name=data_record_path,shard_num=1)

# 定义schema
data_schema = {"file_name": {"type":"string"}, "label":{"type":"int32"}, "data":{"type":"bytes"}}
writer.add_schema(data_schema,"trian_schema")

# 添加数据索引字段
indexes = ["file_name", "label"]
writer.add_index(indexes)

# 数据准备
pneumonia_dir = r"convert_dataset_to_mindrecord/data_to_mindrecord/reasoning/"
pneumonia_file_list = glob.glob(os.path.join(pneumonia_dir,"*.jpeg"))

normal_dir = r"../datasets/convert_dataset_to_mindrecord/images/test/NORMAL/"
normal_file_list = glob.glob(os.path.join(normal_dir,"*.jpeg"))

data = []
for file_name in pneumonia_file_list:
    with open(file_name, "rb") as f:
    bytes_data = f.read()
data.append({"file_name": file_name.split("\\")[1], "label":0, "data":bytes_data})

for file_name in normal_file_list:
with open(file_name, "rb") as f:
    bytes_data = f.read()
data.append({"file_name": file_name.split("\\")[1], "label":1, "data":bytes_data})
# 数据写入
writer.write_raw_data(data)

# 生成本地数据
writer.commit()

```  
• 数据的读取  
```python
    define_data_set = de.MindDataset(file_name,columns_list=['data','label'])  # 读取解析Midecode_opndRecord数据文件构建数据集
     = vision.Decode()
    define_data_set = define_data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=num_parallel_workers)
```
   
### Alex网络的复现
• 使用MindSpore定义神经网络需要继承mindspore.nn.Cell,Cell是所有神经网络（Conv2d等）的基类  
• 神经网络的各层需要预先在__init__方法中定义，然后通过定义construct方法来完成神经网络的前向构造  
• 第一个卷积层用96 9696个大小为11 × 11 × 3 的步长为4 44的卷积核对224 × 224 × 3 的输入图像进行卷积操作然后最大池化；  
第二个卷积层用256 256256个大小为5 × 5 × 48 的步长为1 的卷积核对第一个卷积层的输出进行卷积操作然后最大池化；  
第三个卷积层用384 384384个大小为3 × 3 × 256 的步长为1 的卷积核；  
第四个卷积层用384 384384个大小为3 × 3 × 192 的步长为1 的卷积核；  
第五个卷积层用256 256256个大小为3 × 3 × 192 的步长为1 的卷积核对第四个卷积层的输出进行卷积操作然后最大池化；  
第六个全连接层有4096个神经元；  
第七个全连接层有4096个神经元； 
最后一层是输出层，为1000个分类的softmax层。 总共有6000万个参数。  
 ```python 
self.conv1 = nn.Conv2d(3,96,11,stride=4,pad_mode='valid')
self.conv2 = nn.Conv2d(96, 256, 5, stride=1, pad_mode='same')
self.conv3 = nn.Conv2d(256, 384,3, stride=1, pad_mode='same')
self.conv4 = nn.Conv2d(384, 384, 3, stride=1, pad_mode='same')
self.conv5 = nn.Conv2d(384, 256, 3, stride=1, pad_mode='same')
self.relu = nn.ReLU()
self.max_pool2d = nn.MaxPool2d(kernel_size=3,stride=2)
self.flatten = nn.Flatten()
self.fc1 = nn.Dense(6*6*256,4096)
self.fc2 = nn.Dense(4096,4096)
self.fc3 = nn.Dense(4096,num_classes)
```
• 构建顺序：  
网络顺序：卷积层1->ReLU激活->池化层->卷积层2->ReLU激活->池化层->卷积层3->ReLU激活->卷积层4->ReLU激活->卷积层5->ReLU激活->Flatten多维输入一维化->全连接层1->ReLU激活->全连接层2->ReLU激活->全连接层3。下为实现代码：  
```python
def construct(x):
x = self.conv1(x) #卷积1
x = self.relu(x)  #激活
x = self.max_pool2d(x) #池化

x = self.conv2(x)
x = self.relu(x)
x = self.max_pool2d(x)

x = self.conv3(x)
x = self.relu(x)

x = self.conv4(x)
x = self.relu(x)

x = self.conv5(x)
x = self.relu(x)
x = self.max_pool2d(x)

x = self.flatten(x)

x = self.fc1(x)
x = self.relu(x)

x = self.fc2(x)
x = self.relu(x)

x = self.fc3(x)

return x
 ```
### 服务器&CPU训练模型
• 使用低性能的cpu训练费时费力（x） 

• 使用高效的服务器GPU！（√）  
某次服务器训练结果  
![服务器训练.png](https://s2.loli.net/2022/07/03/OqztPZbB5V9UFHr.png)
### 模型的验证
• 选取数据集中的验证数据部分，使用模型训练产生的.ckpt文件来验证训练好了的模型。  
 调用mindspore.nn下的Accuracy()来计算二分类的正确率
```python 
	
metrics = {"Accuracy": nn.Accuracy()}	

```  
读取验证数据集，并对数据进行同训练数据一样的处理  
```python 

ds_val = create_dateset_val(val_file_name,batch_size,repeat_size)	

```  

调用模型评估接口mindspore.model.eval进行模型的验证，返回为评估结果并输出  
```python 

result = model.eval(ds_val)
print(result)

```
### 模型的推理  
• 加载推理图片，并对图片进行预处理  
```python
def img_pre_process(path):
image = Image.open(path).convert("RGB")
image = image.resize((227, 227))
plt.imshow(image)

# 归一化处理
mean = np.array([122.96757279 / 255, 122.96757279 / 255, 122.96757279 / 255])
std = np.array([55.55022323 / 255, 55.55022323 / 255, 55.55022323 / 255])
image = np.array(image)
image = (image - mean) / std
image = image.astype(np.float32)
# 缩放
image = image / 255
# 图像通道由(h, w, c)转换为(c, h, w)
image = np.transpose(image, (2, 0, 1))
# 扩展数据维数为(1, c, h, w)
image = np.expand_dims(image, axis=0)
return image
```
• 读取checkpoint文件，加载模型  
```python
def reload_model():
#param_dict = load_checkpoint("./check_point/alex-150_79.ckpt")
param_dict = load_checkpoint("alex-4_63.ckpt")
net = AlexNet(num_classes=2)

load_param_into_net(net, param_dict)
net_loss = nn.MSELoss()
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"accuracy"})
return model
```
• 使用模型对图片进行推理，展示结果  
```python
labels = {0:"NORMAL", 1:"PNEUMONIA"}
image1 = r'F:\chest_xray\test\NORMAL\IM-0079-0001.jpeg'
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
model = reload_model()
x1 = img_pre_process(image1)
y1_pre = model.predict(ms.Tensor(x1))
label = np.argmax(y1_pre.asnumpy(), axis=1)
plt.title(labels[label[0]])
plt.show()
```
## 结果说明
• 模型的验证  
![结果1.png](https://s2.loli.net/2022/07/03/iGhdMrgpWou2vfa.png)  
• 模型推理（预测）  
![推理图.png](https://s2.loli.net/2022/07/03/8mzRWDBe1PlCcIJ.png)
## 遇到的问题  
### 如何使用MindSpore框架未定义好的其他数据集  
• 用户可以使用FileWriter API将其他数据集转为mindspore提供的MindRecord数据格式，并且使用MindDateset API加载MindRecord格式的数据集。 MindRecordd是一种高效数据格式。MindRecord的性能优化如下：  
实现多变的用户数据统一存储、访问，训练数据读取更加简便。  
数据聚合存储，高效读取，且方便管理、移动。  
高效的数据编解码操作，对用户透明、无感知。  
可以灵活控制分区的大小，实现分布式训练。  
  
• 数据格式转换与存储：  
可以将每张图片转写为一个字典作为一个数据单元存入Filewriter类型的对象中。字典中可以自定义键值对存储诸如图片二进制数据、图片类别标签、图片文件名等等信息。  
可以选取数据单元中的某几个键作为数据的索引。  
通过FileWriter类别的成员方法commit()将FileWriter类别对象生成.mindrecord文件存储到本地。  

• 数据读取：  
通过mindrecord文件创建MindDataset类别对象，完成对文件的读取。
生成的MindDataset对象是一个迭代器，可以通过对其迭代获取每一个数据单元
数据单元是一个字典类型的对象，通过键索引可以获取其中的数据信息。

具体的代码，前文开发过程中已有详述，在此不再赘述。 
### 如何在MindSpore框架下更高效地训练模型  
将数据集转为mindrecord类型
### 数据的标准化问题  
• 通过图象的标准化，可以提高模型的精度，提高分类准确性。
• 图片标准化后，可以加速模型收敛，并且防止梯度消失与梯度爆炸
• 标准化公式：  
![标准化公式.png](https://s2.loli.net/2022/07/03/YHuFjKy3indstf4.png)  
其中 X 表示原始图像矩阵，X' 表示标准化后的图像矩阵。 μ 表示数据集中图像矩阵的均值向量， σ 表示图像矩阵的标准差向量。
• MindSpore提供的vision模块内的Normalize函数可以对数据集进行归一化处理，只需传输数据集各通道均值与均方差。  
求数据集均值与均方差向量代码如下：  
```python
def channel_mean_std(data: mindspore.dataset.MapDataset):
# 图片数量
img_cnt = 0
# 每张图片的各通道均值的和
sum_mean_rgb = np.array([0, 0, 0], dtype=float)
# 每张图片的各通道均方差的和
sum_d_rgb = np.array([0, 0, 0], dtype=float)
for item in data:
    # 单图均值
    mean_rgb_singleImg = np.array([0, 0, 0], dtype=float)
    # 单图均方差
    d_rgb_singleImg = np.array([0, 0, 0], dtype=float)
    # item[0]是Tensor类数据，使用asnumpy()将其转换为numpy()
    for row in item[0].asnumpy():
        mean_rgb_singleImg += np.mean(row, axis=0) / float(item[0].shape[0])
        d_rgb_singleImg += np.var(row, axis=0) / float(item[0].shape[0])
    sum_mean_rgb += mean_rgb_singleImg
    sum_d_rgb += d_rgb_singleImg
    img_cnt += 1

mean_rgb = sum_mean_rgb / img_cnt
d_rgb = sum_d_rgb / img_cnt
std_rgb = d_rgb ** 0.5
print("mean: ", mean_rgb)
print("std: ", std_rgb)
return mean_rgb, std_rgb

```  
对数据集标准化代码如下：  
```python

normalize_op = CV.Normalize((122.96757279 / 255, 122.96757279 / 255, 122.96757279 / 255), (55.55022323 / 255, 55.55022323 / 255, 55.55022323 / 255))
define_data_set = define_data_set.map(operations=normalize_op,input_columns='data',num_parallel_workers=num_parallel_workers)

```
### 服务器虚拟环境的问题  
• 服务器在重新登录时，在未激活的环境下需要重新激活，但直接conda activate mindspore_py37会报错需要重启，但事实上不需要重启，只需要执行以下指令：  
source activate  
source deactivate  
conda activate mindspore_py37
## 未解决的问题
### 模型的loss值并不是很低  
•  不论是使用cpu还是GPU训练，loss的值都很高，且下降较慢，经过一些参数的调整仍然存在问题，待解决。