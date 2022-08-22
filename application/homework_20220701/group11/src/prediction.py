"""
模型预测
"""
import json
import mindspore.nn as nn
from mindspore import context
import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net
import numpy as np
from PIL import Image
from transformer import Transformer

context.set_context(mode=context.PYNATIVE_MODE)


class CustomWithTestCell(nn.Cell):
    """
    测试网络
    """

    def __init__(self, net):
        super(CustomWithTestCell, self).__init__(auto_prefix=False)
        self.network = net

    def construct(self, data1, data2, data3):
        outputs = self.network(data1, data2, data3)
        pre = (outputs > 0.5).astype(np.float32)

        return pre


network = Transformer(image_size=224, input_channels=3, patch_size=16, embed_dim=64, num_heads=8, num_layers=5,
                      mlp_dim=128, pool='relu')

param_dict = load_checkpoint("model_path/transformer.ckpt")

load_param_into_net(network, param_dict)

idx = 0


def prepare_sequence(seqs, word_to_idx):
    """
    word2id
    """
    seq_outputs, seq_length = [], []
    # print(seqs)
    max_len = max([len(i) for i in seqs])
    print(max_len)

    for seq in seqs:
        # print(seq)
        seq_length.append(len(seq))
        idxs = [word_to_idx[w] for w in seq]
        idxs.extend([word_to_idx['<pad>'] for i in range(max_len - len(seq))])
        # print('idsx',len(idxs),'seq_length',seq_length)
        seq_outputs.append(idxs)

    return np.array(seq_outputs, dtype=np.int64), np.array(seq_length, dtype=np.int64)


sen_data = []
img_list = []
label = []

idx = 1
with open('dataset/test/test.txt') as f:
    for line in f:
        line = line.strip()
        parts = line.split('\t')
        sen_data.append(parts[0].split(' '))
        img_list.append(parts[1])
        label.append(int(parts[2]))
        if idx == 10:
            break
        idx += 1

img = []

idx = 1
for img_path in img_list:
    image = Image.open(img_path)
    image = image.resize((224, 224))
    x = np.copy(image)
    data = ((x - np.min(x)) / (np.max(x) - np.min(x))).astype(np.float32)
    # print(data.shape)
    data = data.swapaxes(0, 2)
    data = data.swapaxes(1, 2)
    # print(data.shape)
    img.append(data)
    if idx == 10:
        break
    idx += 1

with open('dataset/word2id.json') as f:
    word2id = json.load(f)

sen, length = prepare_sequence(sen_data, word2id)

image = ms.Tensor(np.array(img, dtype=np.float32))
sentence = ms.Tensor(sen)
length = ms.Tensor(length)
label = ms.Tensor(np.array(label))

custom_test_net = CustomWithTestCell(network)
custom_test_net.set_train(False)

predict = custom_test_net(image, sentence, length).reshape(-1)

for i in range(len(predict)):
    print(f"文本：{' '.join(sen_data[i])},真实值：{label[i]},预测值:{predict[i]}")
