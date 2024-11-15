from download import download
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from SING import SING

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

# hyperparameter
input_size = 784
hidden_size = 128
output_size = 10
lr = 0.0001
num_epochs = 3


# Download data from open datasets

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)


def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

# Define model


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
# print(model)

loss_fn = nn.CrossEntropyLoss()
# optimizer=nn.Adam(model.trainable_params(),learning_rate=Tensor(1e-2,ms.float32))
# print(model.trainable_params())
optimizer = SING(model.trainable_params(),
                 learning_rate=Tensor(1e-3, ms.float32))

loss_net = nn.WithLossCell(model, loss_fn)
train_net = nn.TrainOneStepCell(loss_net, optimizer)


def train(model, dataset):
    size = dataset.get_dataset_size()

    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):

        loss = train_net(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test(model, dataset, loss_fn):
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
    print(
        f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)


for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)
    test(model, test_dataset, loss_fn)
print("Done!")
