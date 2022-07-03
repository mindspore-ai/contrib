"""
    module
"""
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore.train.callback import TimeMonitor
from mindspore import context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import export
from mindspore import Tensor
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.engine.callback import ValAccMonitor
from mindvision.classification.models import efficientnet_b0


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
num_epochs = 3
classindex = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "Nothing": 10}

def create_dataset(path, batch_size=32, train=True, image_size=224):
    """
    create dataset
    """
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
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_path = "./My_Hand_Gesture_Dataset/Training_Data"
dataset_train = create_dataset(train_path, train=True)
val_path = "./My_Hand_Gesture_Dataset/Testing_Data"
dataset_val = create_dataset(val_path, train=False)

# 使用efficientnet_b0网络作为分类器
network = efficientnet_b0(num_classes=11, pretrained=False)
# 使用Adam优化器训练模型
network_opt = nn.Adam(params=network.trainable_params())
network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=11)
metrics = {"Accuracy": nn.Accuracy()}
model = ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

model.train(num_epochs, dataset_train, callbacks=[ValAccMonitor(model, dataset_val, num_epochs), TimeMonitor()])

# 定义并加载网络参数
net = efficientnet_b0(num_classes=11)
param_dict = load_checkpoint("best.ckpt")
load_param_into_net(net, param_dict)
# 将模型由ckpt格式导出为MINDIR格式
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
export(net, Tensor(input_np), file_name="efficientnet", file_format="MINDIR")
print("model transform success.......................")
