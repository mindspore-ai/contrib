"""
加载数据集
"""
import json
import mindspore as ms
from mindspore.common.initializer import Normal
import mindspore.dataset as ds
import numpy as np
from PIL import Image

np.random.seed(1024)


def get_train_data():
    x = np.random.randint(0, 255, [200, 224, 224, 3])
    mydata = (x - np.min(x)) / (np.max(x) - np.min(x))

    y = ms.Tensor(shape=(200, 30, 768), dtype=ms.float32, init=Normal())

    z = np.random.randint(0, 2, (200, 1))

    return ms.Tensor(mydata), ms.Tensor(y), ms.Tensor(z)


def prepare_sequence(seqs, word_to_idx):
    """文本转id"""
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


class MyDataset:
    """自定义数据集类"""

    def __init__(self, flag='train'):
        """自定义初始化操作"""
        # self.image,self.sentence,self.label = get_train_data()  # 自定义数据
        # print(self.image,self.sentence.self.label)
        with open('dataset/word2id.json') as f:

            word2id = json.load(f)

        sen_data = []
        img_list = []
        label = []

        if flag == 'train':
            with open('dataset/train/train.txt') as f:
                for line in f:
                    line = line.strip()
                    parts = line.split('\t')
                    sen_data.append(parts[0].split(' '))
                    img_list.append(parts[1])
                    label.append(int(parts[2]))
        elif flag == 'val':
            with open('dataset/val/val.txt') as f:
                for line in f:
                    line = line.strip()
                    parts = line.split('\t')
                    sen_data.append(parts[0].split(' '))
                    img_list.append(parts[1])
                    label.append(int(parts[2]))
        elif flag == 'test':
            with open('dataset/test/test.txt') as f:
                for line in f:
                    line = line.strip()
                    parts = line.split('\t')
                    sen_data.append(parts[0].split(' '))
                    img_list.append(parts[1])
                    label.append(int(parts[2]))

        sen, length = prepare_sequence(sen_data, word2id)

        img = []
        for img_path in img_list:
            image = Image.open(img_path)
            image = image.resize((224, 224))
            x = np.copy(image)
            img_data = ((x - np.min(x)) / (np.max(x) - np.min(x))).astype(np.float32)
            # print(data.shape)
            img_data = img_data.swapaxes(0, 2)
            img_data = img_data.swapaxes(1, 2)
            # print(data.shape)
            img.append(img_data)

        # x = np.random.randint(0, 255, [200, 3,224,224])
        # data = ((x - np.min(x)) / (np.max(x) - np.min(x))).astype(np.float32)
        #
        # y = ms.Tensor(shape=(200, 30, 32), dtype=ms.float32, init=Normal()).asnumpy()
        #
        # z = np.random.randint(0, 2, (200,1)).astype(np.float32)

        # print(len(img))
        self.image = np.array(img, dtype=np.float32)
        self.sentence = sen
        self.length = length
        self.label = np.array(label, dtype=np.float32).reshape(-1, 1)

    def __getitem__(self, index):

        """自定义随机访问函数
        实际的代码 return self.image[index], self.sentence[index], self.length[index], self.label[index]
        """

        return self.image[index], self.sentence[index], self.length[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self.image)


if __name__ == '__main__':
    # 实例化数据集类
    dataset_generator = MyDataset('val')
    dataset = ds.GeneratorDataset(dataset_generator, ["image", "sentence", "length", "label"], shuffle=False).batch(50)

    # 迭代访问数据集
    for data in dataset.create_dict_iterator():
        data1 = data['image'].shape
        data2 = data['sentence'].shape
        data3 = data['length'].shape
        label1 = data['label'].shape
        print(data1, data2, data3, label1)

    # 打印数据条数
    print("data size:", dataset.get_dataset_size())
