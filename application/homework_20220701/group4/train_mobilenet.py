"""
    module
"""
import numpy as np
from mindvision.classification.models import mobilenet_v2
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.engine.callback import ValAccMonitor
import mindspore as ms
import mindspore.nn as nn
from mindspore import export
from mindspore import Tensor
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore.train.callback import TimeMonitor
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision


# 选择执行模式为图模式；指定训练使用的平台为"GPU"，如需使用昇腾硬件可将其替换为"Ascend"
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
num_epochs = 5


def create_dataset(path, batch_size=32, train=True, image_size=224):
    """
    create dataset
    """
    classindex = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "Nothing": 10}
    dataset = ds.ImageFolderDataset(path, num_parallel_workers=4, class_indexing=classindex)
    # 图像增强操作
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if train:
        trans = [
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        trans = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    # 设置batch_size的大小，若最后一次抓取的样本数小于batch_size，则丢弃
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

# 加载训练数据集
train_path = "./My_Hand_Gesture_Dataset/Training_Data"
dataset_train = create_dataset(train_path, train=True)
# 加载验证数据集
val_path = "./My_Hand_Gesture_Dataset/Testing_Data"
dataset_val = create_dataset(val_path, train=False)

# 创建模型,其中目标分类数为2，图像输入大小为(224,224)
network = mobilenet_v2(num_classes=11, resize=224)
# 模型参数存入到param_dict
param_dict = ms.load_checkpoint("./mobilenet_v2_1.0_224.ckpt")
# 获取mobilenet_v2网络最后一个卷积层的参数名
filter_list = [x.name for x in network.head.classifier.get_parameters()]
# 删除预训练模型的最后一个卷积层


def filter_ckpt_parameter(origin_dict, param_filter):
    """filter model"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

filter_ckpt_parameter(param_dict, filter_list)
# 加载预训练模型参数作为网络初始化权重
ms.load_param_into_net(network, param_dict)
# 定义优化器
network_opt = nn.Momentum(params=network.trainable_params(), learning_rate=0.01, momentum=0.9)
# network_opt = nn.Adam(params=network.trainable_params())
# 定义损失函数
network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=11)
# network_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义评价指标
metrics = {"Accuracy": nn.Accuracy()}
# 初始化模型
model = ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)
# 模型训练与验证，训练完成后保存验证精度最高的ckpt文件（best.ckpt）到当前目录下
print("start training model...................................")
model.train(num_epochs, dataset_train, callbacks=[ValAccMonitor(model, dataset_val, num_epochs), TimeMonitor()])

# 定义并加载网络参数
net = mobilenet_v2(num_classes=11, resize=224)
param_dict = load_checkpoint("best.ckpt")
load_param_into_net(net, param_dict)
# 将模型由ckpt格式导出为MINDIR格式
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
export(net, Tensor(input_np), file_name="mobilenet_v2_1.0_224", file_format="MINDIR")
print("model transform success.......................")
