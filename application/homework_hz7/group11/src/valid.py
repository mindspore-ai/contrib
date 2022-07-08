"""模型验证"""

import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context
from mindspore.ops import operations as P
from mindspore import load_checkpoint, load_param_into_net
import numpy as np
from transformer import Transformer
from dataload import MyDataset

context.set_context(mode=context.PYNATIVE_MODE)

network = Transformer(image_size=224, input_channels=3, patch_size=16, embed_dim=64, num_heads=8, num_layers=5,
                      mlp_dim=128, pool='relu')

dataset_generator = MyDataset()
train_dataset = ds.GeneratorDataset(dataset_generator, ["image", "sentence", "length", "label"], shuffle=False).batch(
    64)

param_dict = load_checkpoint("model_path/transformer.ckpt")

load_param_into_net(network, param_dict)


class CustomWithEvalCell(nn.Cell):
    """
    验证网络
    """

    def __init__(self, net):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = net

    def construct(self, data1, data2, data3, label):
        outputs = self.network(data1, data2, data3)
        x = (outputs > 0.5).astype(np.float32)
        acc = (x == label).astype(np.float32)
        mean = P.ReduceMean()
        acc = mean(acc)
        return acc


custom_eval_net = CustomWithEvalCell(network)
custom_eval_net.set_train(False)

# 定义训练网络，封装网络和优化器
train_net = CustomWithEvalCell(network)
# 设置网络为训练模式
train_net.set_train(False)

# 真正训练迭代过程
steps = train_dataset.get_dataset_size()
step = 0

acc_avg = 0
for d in train_dataset.create_dict_iterator():
    print(d['image'].shape, d['sentence'].shape, d["length"].shape, d['label'].shape)
    result = train_net(d["image"], d["sentence"], d["length"], d["label"])
    acc_avg = result + acc_avg
    print(f"step: [{step} / {steps}], "
          f"ACC: {result}")
    step = step + 1

print(f"acc_avg:{acc_avg / step}")
