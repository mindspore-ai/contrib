import mindspore as ms
from mindspore import nn, dataset as ds, Model, context
from mindvision.dataset import Mnist
from mindspore.train.callback import Callback
import math
from tsgd import TSGD

# 设置mindspore随机种子
ms.set_seed(0)

# 1. 设置训练环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # 如果有GPU支持，可以将设备设置为"Ascend"

# 2. 参数设置
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# 3. 加载MNIST数据集
def create_dataset(data_path, batch_size=64, training=True):
    mnist_ds = Mnist(path=data_path, split="train" if training else "test", batch_size=batch_size, resize=28, download=False).run()
    return mnist_ds

train_dataset = create_dataset("./datasets/MNIST_Data", batch_size=batch_size, training=True)
test_dataset = create_dataset("./datasets/MNIST_Data", batch_size=batch_size, training=False)

# 4. 定义FFN网络
class FeedforwardNeuralNet(nn.Cell):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(FeedforwardNeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Dense(input_size, hidden_size, weight_init=Normal(0.02))
        self.fc1 = nn.Dense(input_size, hidden_size)

        self.relu = nn.ReLU()
        # self.fc2 = nn.Dense(hidden_size, num_classes, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(hidden_size, num_classes)

    def construct(self, x):
        # print(x)
        x = self.flatten(x)
        # print(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = FeedforwardNeuralNet()

# 5. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()

# 计算 iters
trainSampleSize = 60000     # MNIST数据集的训练集包含 60,000 张手写数字图像
iters = math.ceil(trainSampleSize / batch_size) * num_epochs

# 构建优化器
# optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)
# optimizer = nn.Adagrad(model.trainable_params(), learning_rate=learning_rate)
optimizer = TSGD(model.trainable_params(), iters=iters)


# 6. 训练模型
# 自定义回调函数计算平均Loss
class AverageLossCallback(Callback):
    def __init__(self):
        self.epoch_loss = 0
        self.steps = 0

    def epoch_begin(self, run_context):
        # 每个 epoch 开始时重置 loss 和步数计数
        self.epoch_loss = 0
        self.steps = 0

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.epoch_loss += cb_params.net_outputs.asnumpy()
        self.steps += 1

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        avg_loss = self.epoch_loss / self.steps
        print(f"Epoch [{cb_params.cur_epoch_num}], Average Loss: {avg_loss:.4f}")

def train(model, train_dataset, loss_fn, optimizer, num_epochs):
    model = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    print("Training...")
    model.train(num_epochs, train_dataset, callbacks=[AverageLossCallback()], dataset_sink_mode=False)

# 7. 测试模型
def evaluate(model, test_dataset):
    model = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    print("Evaluating...")
    acc = model.eval(test_dataset, dataset_sink_mode=False)
    print(f"Test Accuracy: {acc['accuracy'] * 100:.2f}%")

# 8. 执行训练和评估
train(model, train_dataset, loss_fn, optimizer, num_epochs)
evaluate(model, test_dataset)
