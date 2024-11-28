import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.dataset import transforms
import mindspore.dataset.vision as tools
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import context
from PIL import Image
import numpy as np
import os
import random
import kagglehub
import tqdm
import matplotlib.pyplot as plt
import ReNet



# 下载数据集
path = kagglehub.dataset_download("ztaihong/weizmann-horse-database")+"\\weizmann_horse_db"
print("Path to dataset files:", path)

# 设置运行模式为图模式
context.set_context(mode=context.GRAPH_MODE)
# 定义超参数
batch_size = 4
learning_rate = 0.005
num_epochs = 4
input_size = (32, 32)  # 输入图像大小



# 定义数据预处理
transform = transforms.Compose([
    tools.Resize(input_size),
    tools.ToTensor()
])

# 自定义数据集类，用于加载图像和对应的掩码数据
class SegmentationDataset:
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.image_files = os.listdir(os.path.join(root_dir, "horse"))
        self.mask_files = os.listdir(os.path.join(root_dir, "mask"))
        self.images = []
        self.masks = []
        for index in tqdm.tqdm(range(len(self.image_files))):
            image_path = os.path.join(self.root_dir, "horse", self.image_files[index])
            mask_path = os.path.join(self.root_dir, "mask", self.mask_files[index])

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                image = self.transform(image)[0]
                mask = np.array(self.transform(mask)[0]*255,dtype="float32")
                

            self.images.append(image)
            self.masks.append(mask)
        
        
    def __getitem__(self, index):
        
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.image_files)


# 创建训练集实例
train_dataset = SegmentationDataset(root_dir=path, transform=transform, is_train=True)
# 创建训练集数据加载器
data_loader = ds.GeneratorDataset(source=train_dataset, column_names=['image','mask']).batch(batch_size).split([0.7,0.3])
train_data_loader = data_loader[0].create_dict_iterator()
test_data_loader = data_loader[1].create_dict_iterator()



# 创建模型实例、损失函数和优化器
model = ReNet.ReSeg()
criterion = nn.BCELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)


def forward_fn(x, y):
    z = model(x)
    loss = criterion(z, y)
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, weights=model.trainable_params())


# 定义评估指标计算函数
def calculate_metrics(outputs, masks):
    preds = ops.where(outputs > 0.5, mindspore.Tensor(1, dtype=outputs.dtype), mindspore.Tensor(0, dtype=outputs.dtype))
    preds = preds.view(-1)
    masks = masks.view(-1)

    correct_pixels = ops.equal(preds, masks).sum()
    total_pixels = preds.size

    pixel_accuracy = 1.0 * correct_pixels / total_pixels

    return pixel_accuracy

# 训练模型
for epoch in range(num_epochs):
    model.set_train(True)
    train_loss_total = 0
    train_accuracy_total = 0
    i = 0
    print(f"begin epoch:{epoch}")
    for item in train_data_loader:
        i+=1
        images = item["image"]
        masks = item["mask"]
        
        loss, grads = grad_fn(images, masks)
        optimizer(grads)

        train_loss_total += loss.item()

        if(i%20==0):
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item()}')

# 在测试集上评估模型
model.set_train(False)
test_accuracy_total = 0
i = 0
for item in test_data_loader:
    outputs = model(item["image"])
    test_accuracy = calculate_metrics(outputs, item["mask"])
    i += 1
    test_accuracy_total = test_accuracy + test_accuracy_total

test_accuracy_avg = test_accuracy_total / i
print(f'Test Accuracy: {test_accuracy_avg}')

# 展示
for item in test_data_loader:
    outputs = model(item["image"])[0]
    image = item['image'][0]
    label = item['mask'][0]
   
    image = image.asnumpy()
    mask = outputs.asnumpy()
    label = label.asnumpy()
    mask = np.repeat(mask, 3, axis=0)
    label = np.repeat(label, 3, axis=0)

    image = np.transpose(image, (1, 2, 0))
    mask = np.transpose(mask, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image 1")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("mask")

    plt.subplot(1, 3, 3)
    plt.imshow(label)
    plt.title("label")

    plt.show()
    break