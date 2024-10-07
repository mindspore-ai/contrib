from fedEnsemble_ms import train
from models_ms import BaseModel
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from download import download
from mindspore.dataset import vision,transforms
import os


class CustomDataGenerator:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 假设数据格式为 (input, label)
        input_data, label = self.data[index]
        return input_data, label
def load_cifar10_dataset(data_dir, usage='train'):
    # 加载 CIFAR-10 数据集
    cifar10_dataset = ds.Cifar10Dataset(dataset_dir=data_dir, usage=usage)

    # 定义图像变换
    image_transform = [
        vision.Resize((32, 32)),
        # vision.ToTensor(),
        vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        vision.HWC2CHW()
    ]

    cifar10_dataset = cifar10_dataset.map(operations=image_transform, input_columns=["image"])

    return cifar10_dataset

def main():
    # Set Fed-Ensemble hyperparameters
    NUM_MODELS = 5
    NUM_CLIENTS = 100
    NUM_STRATA = 10
    NUM_SELECTED_CLIENTS = 5
    NUM_AGES = 50
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001

    hidden_size = 100
    num_classes = 10

    # 定义数据集URL
    url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"
    if not os.path.exists("./data/cifar-10-batches-bin"):
        download(url, "./data", kind="tar.gz", replace=True)

    data_dir = './data/cifar-10-batches-bin'

    # 加载 CIFAR-10 训练数据集
    train_dataset = load_cifar10_dataset(data_dir, usage='train')

    # 加载 CIFAR-10 测试数据集
    validation_dataset = load_cifar10_dataset(data_dir, usage='test')
    # Create model architecture
    network = BaseModel(hidden_size, num_classes)

    # Perform Fed-Ensemble in federated learning
    train(NUM_CLIENTS, NUM_MODELS, NUM_SELECTED_CLIENTS, NUM_AGES, BaseModel, train_dataset,
                                   validation_dataset, batch_size, num_epochs, learning_rate, NUM_STRATA)

if __name__ == '__main__':
    main()
