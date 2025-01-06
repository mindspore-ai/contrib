import mindspore as ms
from mindspore import Tensor, nn, context
from mindspore.train import Model
from mindspore.train.callback import LossMonitor

# 设置MindSpore的执行上下文
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 定义模型
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(784, 128)
        self.fc2 = nn.Dense(128, 64)
        self.fc3 = nn.Dense(64, 10)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义损失函数和优化器
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)

# 创建模型
model = Model(net, loss_fn, optimizer, metrics={'accuracy'})

# 训练模型
model.train(epoch=5, train_dataset=train_dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)

# 评估模型
acc = model.eval(test_dataset, dataset_sink_mode=False)
print("Accuracy:", acc)
