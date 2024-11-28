import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
import numpy as np
from download import download
# 引入AutomaticWeightedLoss
from AutomaticWeightedLoss import AutomaticWeightedLoss


# 数据准备
url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe('MNIST_Data/train', batch_size=64)
test_dataset = datapipe('MNIST_Data/test', batch_size=64)


# 定义神经网络模型
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()


# 定义超参、损失函数和优化器
epochs = 3
batch_size = 64
learning_rate = 1e-2

# 创建损失函数
loss_fn1 = nn.CrossEntropyLoss()  # 第一个损失函数
loss_fn2 = nn.CrossEntropyLoss()  # 第二个损失函数，为简易起见，将二者设为相同

# AutomaticWeightedLoss实例
awl = AutomaticWeightedLoss(2)  # 有两个损失函数

# 定义优化器
optimizer = nn.SGD(params=[
                {'params': model.trainable_params()},
                {'params': awl.trainable_params(), 'weight_decay': 0}
            ], learning_rate=learning_rate)

# 训练与评估
# 定义前向传播函数，包含多任务损失
def forward_fn(data, label):
    logits = model(data)
    loss1 = loss_fn1(logits, label)
    loss2 = loss_fn2(logits, label)
    # 使用 AutomaticWeightedLoss 计算总损失
    loss_sum = awl(loss1, loss2)
    return loss_sum, logits

# 获取梯度函数
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 定义训练步骤
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train_loop(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

def test_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 训练与评估循环
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataset)
    test_loop(model, test_dataset, loss_fn1)  # 这里只测试第一个损失函数
print("Done!")
