import mindspore as ms
from mindspore import nn, dataset as ds, Model, context
from mindvision.dataset import Mnist
from mindspore.train.callback import Callback
from GaussianBlock import GaussianBlock

# 设置mindspore随机种子
ms.set_seed(0)

# 1. 设置训练环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU") 

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

# 4. 定义使用GaussianBlock的简单网络
norm_axes = [1, 1, 1]  # Axes for each layer in the GaussianBlock.
num_heads = [4, 4, 4]  # Number of attention heads for each layer.
num_gaussians = [5, 5, 5]  # Number of Gaussians per head for each layer.
num_layers = 3  # Total number of layers in the GaussianBlock.
padding_value = None  # Padding value for sequences in the input tensor.
eps = 1e-8  # Small epsilon value for numerical stability.

# Initialize the GaussianBlock
attention_block = GaussianBlock(norm_axes, num_heads, num_gaussians, num_layers, padding_value, eps)

# Example neural network with GaussianBlock
class SimpleNetworkWithGaussianBlock(nn.Cell):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(SimpleNetworkWithGaussianBlock, self).__init__()
        # Initialize GaussianBlock for attention mechanism
        self.attention_block = attention_block
        # Initialize a linear layer
        self.linear = nn.Dense(input_dim, output_dim)

    def construct(self, x):
        x = x.view(-1, 28*28)  # 将图像展平为1维向量
        # Apply GaussianBlock for attention
        x = self.attention_block(x)
        # Apply linear layer
        x = self.linear(x)
        return x


# 定义FFN网络
# class FeedforwardNeuralNet(nn.Cell):
#     def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
#         super(FeedforwardNeuralNet, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Dense(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Dense(hidden_size, num_classes)

#     def construct(self, x):
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# model = FeedforwardNeuralNet()
model = SimpleNetworkWithGaussianBlock()

# 5. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()

# 构建优化器
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

# 6. 训练模型
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
