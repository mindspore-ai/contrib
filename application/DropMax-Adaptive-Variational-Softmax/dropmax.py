# 完整的 MindSpore 代码，包括数据下载、解压、加载、模型定义、训练和测试

import os
import zipfile
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import requests

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, dtype as mstype, context
from mindspore.dataset import MnistDataset
from mindspore.dataset import transforms as C
from mindspore.dataset.vision import Resize, Normalize, HWC2CHW
import mindspore.ops as ops

from mindspore.dataset.vision import Inter

def download_and_extract(url, download_path, extract_path, kind='zip'):
    """
    下载并解压文件。
    """
    filename = url.split('/')[-1]
    file_path = os.path.join(download_path, filename)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {filename}.")
    print(f"Extracting {filename}...")
    if kind == 'zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif kind == 'tar.gz':
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
    print(f"Extracted {filename}.")

mnist_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
mnist_download_path = "./data"
mnist_extract_path = "./data/MNIST"

# download_and_extract(mnist_url, mnist_download_path, mnist_extract_path, kind="zip")

expected_files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
]

missing_files = [f for f in expected_files if not os.path.isfile(os.path.join(mnist_extract_path, f))]

if missing_files:
    print(f"Missing MNIST files: {missing_files}")
else:
    print("All MNIST files are present.")

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  # 如果有 GPU，可以设置为 "GPU"

def dropmax(y, z, dim=1):
    e = 1e-20
    exp_op = ops.Exp()
    sum_reduce = ops.ReduceSum(keep_dims=True)
    numerator = (z + e) * exp_op(y)
    denominator = sum_reduce(numerator, axis=dim)
    return numerator / denominator

class BaseCNN(nn.Cell):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, pad_mode='valid')  # 输出: [batch, 32, 30, 30]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, pad_mode='valid') # 输出: [batch, 64, 28, 28]
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: [batch, 64, 14, 14]
        self.flatten = ops.Flatten()
        self.fc1 = nn.Dense(14 * 14 * 64, 128)
        self.fc2 = nn.Dense(128, 10)

    def construct(self, x):
        x = self.relu(self.conv1(x))         # [batch, 32, 30, 30]
        x = self.relu(self.conv2(x))         # [batch, 64, 28, 28]
        x = self.max_pool(x)                  # [batch, 64, 14, 14]
        x = self.flatten(x)                   # [batch, 14*14*64]
        x = self.relu(self.fc1(x))           # [batch, 128]
        x = self.fc2(x)                       # [batch, 10]
        return x

class CNN(nn.Cell):
    def __init__(self):
        super(CNN, self).__init__()
        self.parameterize = BaseCNN()
        self.classifier = BaseCNN()
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.clip = ops.clip_by_value

    def construct(self, x):
        y = self.classifier(x)                # [batch, 10]
        p = self.parameterize(x)              # [batch, 10]
        tau = 1e-1

        minval = Tensor(0.0, mstype.float32)
        maxval = Tensor(1.0, mstype.float32)
        u = ops.uniform(p.shape, minval, maxval, seed=5)     # [batch, 10]
        p = self.clip(p, 1e-7, 1 - 1e-7)
        u = self.clip(u, 1e-7, 1 - 1e-7)

        term1 = (1 / tau) * self.log(p)                      # [batch, 10]
        term2 = self.log(1 - p)                              # [batch, 10]
        term3 = self.log(u)                                  # [batch, 10]
        term4 = self.log(1 - u)                              # [batch, 10]
        z = self.sigmoid(term1 - term2 + term3 - term4)     # [batch, 10]
        output = self.log(dropmax(y, z, dim=1))             # [batch, 10]
        return output

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, n_epoch, lr):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epoch = n_epoch
        self.optimizer = nn.SGD(self.model.trainable_params(), learning_rate=lr)
        self.loss_fn = nn.NLLLoss()
        self.train_net = nn.WithLossCell(self.model, self.loss_fn)
        self.train_step = nn.TrainOneStepCell(self.train_net, self.optimizer)
        self.train_step.set_train()

    def train(self):
        for epoch in range(self.n_epoch):
            print(f"Starting Epoch {epoch + 1}")
            epoch_loss = 0.0
            step = 0
            for data in self.train_dataset.create_dict_iterator():
                images = data["image"]
                labels = data["label"]
                loss = self.train_step(images, labels)
                epoch_loss += loss.asnumpy()
                step += 1
                if step % 100 == 0:
                    print(f"  Step {step}, Loss: {loss.asnumpy():.4f}")
            avg_loss = epoch_loss / step
            print(f"Epoch: {epoch + 1} Average Loss: {avg_loss:.4f}")

    def test(self):
        self.model.set_train(False)
        correct = 0
        total = 0
        argmax = ops.ArgMaxWithValue(axis=1)
        for data in self.test_dataset.create_dict_iterator():
            images = data["image"]
            labels = data["label"]
            preds = self.model(images)
            predicted = argmax(preds)[1]
            correct += (predicted == labels).sum().asnumpy()
            total += labels.shape[0]
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

# 数据预处理
# 定义图像转换和标签转换
transform_img = [
    Resize((32, 32), interpolation=Inter.LINEAR),
    HWC2CHW(),
    Normalize(mean=(0.5,), std=(0.5,)),
    C.TypeCast(mstype.float32)
]

transform_label = [
    C.TypeCast(mstype.int32)
]

mnist_data_dir = "./data/MNIST"

if not os.path.exists(mnist_data_dir):
    raise FileNotFoundError(f"MNIST data directory not found: {mnist_data_dir}")

missing_files = [f for f in expected_files if not os.path.isfile(os.path.join(mnist_data_dir, f))]
if missing_files:
    raise FileNotFoundError(f"Missing MNIST files in {mnist_data_dir}: {missing_files}")

train_dataset = MnistDataset(mnist_data_dir, usage='train')
train_dataset = train_dataset.map(operations=transform_img, input_columns="image")
train_dataset = train_dataset.map(operations=transform_label, input_columns="label")
train_dataset = train_dataset.batch(96, drop_remainder=True)

test_dataset = MnistDataset(mnist_data_dir, usage='test')
test_dataset = test_dataset.map(operations=transform_img, input_columns="image")
test_dataset = test_dataset.map(operations=transform_label, input_columns="label")
test_dataset = test_dataset.batch(32, drop_remainder=False)

cnn_model = CNN()
trainer = Trainer(
    model=cnn_model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    n_epoch=5,
    lr=0.001
)

trainer.train()

trainer.test()

# 可视化部分样本
def plot(imgs, first_origin=None):
    num_rows = 1
    num_cols = len(imgs)

    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for idx, img in enumerate(imgs):
        ax = axs[0, idx]
        img_np = img.asnumpy().squeeze() * 0.5 + 0.5
        ax.imshow(img_np, cmap='gray')
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if first_origin:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    plt.tight_layout()
    plt.show()

# 显示部分训练样本
images = []
labels = []
for data in train_dataset.create_dict_iterator():
    images.append(data["image"])
    labels.append(data["label"])
    if len(images) >= 5:
        break

plot(images[:5])