"""
模型训练
"""

import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
import mindspore.ops as ops
import mindspore as ms
from transformer import Transformer
from dataload import MyDataset

vocab_size = 10310

context.set_context(mode=context.PYNATIVE_MODE)

network = Transformer(image_size=224, input_channels=3, patch_size=16, embed_dim=32, num_heads=8, num_layers=3,
                      mlp_dim=128, pool='relu')

# 定义损失函数
net_loss = nn.BCELoss(reduction='mean')

# 定义优化器函数
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)

# # 设置模型保存参数，模型训练保存参数的step为1875
# config_ck = CheckpointConfig(save_checkpoint_steps=500, keep_checkpoint_max=10)
#
# # 应用模型保存参数
# ckpoint = ModelCheckpoint(prefix="multi_model", directory="./vit_bert", config=config_ck)


dataset_generator = MyDataset()
train_dataset = ds.GeneratorDataset(dataset_generator, ["image", "sentence", "length", "label"], shuffle=False).batch(
    50)


class WithLossCellTrans(nn.Cell):
    def __init__(self, net, loss_fn):
        super(WithLossCellTrans, self).__init__(auto_prefix=True)
        self.net = net
        self.loss_fn = loss_fn

    def construct(self, image, sentence, seq_length, label):
        out = self.net(image, sentence, seq_length)
        loss = self.loss_fn(out, label)
        return loss.mean()


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, net, optimizer):
        """入参有两个：训练网络，优化器"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = net  # 定义前向网络
        self.network.set_grad()  # 构建反向网络
        self.optimizer = optimizer  # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, image, sentence, seq_length, label):
        loss = self.network(image, sentence, seq_length, label)  # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(image, sentence, seq_length, label)  # 进行反向传播，计算梯度
        self.optimizer(grads)  # 使用优化器更新权重参数
        return loss


# 连接前向网络与损失函数
net_with_loss = WithLossCellTrans(network, net_loss)
opt = nn.Momentum(network.trainable_params(), learning_rate=0.005, momentum=0.9)

# 定义训练网络，封装网络和优化器
train_net = CustomTrainOneStepCell(net_with_loss, opt)
# 设置网络为训练模式
train_net.set_train()

# 真正训练迭代过程
step = 0
epochs = 2
steps = train_dataset.get_dataset_size()

for epoch in range(epochs):
    for d in train_dataset.create_dict_iterator():
        print(d['image'].shape, d['sentence'].shape, d["length"].shape, d['label'].shape)
        result = train_net(d["image"], d["sentence"], d["length"], d["label"])
        print(f"Epoch: [{epoch} / {epochs}], "
              f"step: [{step} / {steps}], "
              f"loss: {result}")
        step = step + 1

ms.save_checkpoint(network, 'model_path/transformer.ckpt')
