import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


mindspore.set_device('CPU')

class InverseNet(nn.Cell):
    def __init__(self, l):
        super(InverseNet, self).__init__()
 
        self.an = nn.SequentialCell([
            nn.Conv2d(3, 64, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU()
        ])
        feats = self.an
        cl = []
        if l < 5:
            self.conv_lin = 'conv'
            split_point = 0
            conv_cntr = 0
            for lay in feats:
                if isinstance(lay, nn.Conv2d):
                    conv_cntr += 1
                split_point += 1

                if conv_cntr == l:
                    break

            self.an = nn.SequentialCell(list(feats)[:split_point + 1])

        elif 6 <= l <= 8:
            self.conv_lin = 'lin'
            split_point = 0
            lin_cntr = 5
            for lay in cl:
                if isinstance(lay, nn.Dense):
                    lin_cntr += 1
                split_point += 1

                if lin_cntr == l:
                    break

            self.an = nn.SequentialCell(list(cl)[:split_point + 1])

        # Freeze base network parameters
        for param in self.an.get_parameters():
            param.requires_grad = False

        self.lin_net1 = nn.SequentialCell([
            nn.Dense(1000, 4096),
            nn.LeakyReLU(0.2),
            nn.Dense(4096, 4096),
            nn.LeakyReLU(0.2),
            nn.Dense(4096, 4096),
            nn.LeakyReLU(0.2)
        ])

        self.lin_net2 = nn.SequentialCell([
            nn.Conv2dTranspose(256, 256, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2dTranspose(256, 128, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2dTranspose(128, 64, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2dTranspose(64, 32, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2dTranspose(32, 3, 8, stride=2),
            nn.LeakyReLU(0.2)
        ])

    def construct(self, x):
        x = self.an(x)
        if self.conv_lin == 'conv':
            pass
        elif self.conv_lin == 'lin':

            flatten = nn.Flatten()
            x = flatten(x)
            adjust_dim = nn.Dense(x.shape[-1], 1000)
            x = adjust_dim(x)
            x = self.lin_net1(x)
            x = x.view(x.shape[0], 256, 4, 4)
            x = self.lin_net2(x)
            # 调整输出形状以匹配输入形状
            resize = ops.ResizeBilinearV2()
            # 修改此处，将输出调整为输入的形状
            x = resize(x, (224, 224))
        return x


def main():
    epochs = 5  # 增加训练轮数
    batch_size = 16  # 增加批次大小
    num_batches = 20  # 增加批次数量

    # 模拟输入数据
    input_data = Tensor(np.random.randn(num_batches * batch_size, 3, 224, 224).astype(np.float32))

    model = InverseNet(8)
    criterion = nn.MSELoss()
    optimizer = nn.Adam(filter(lambda x: x.requires_grad, model.get_parameters()))

    net_with_loss = nn.WithLossCell(model, criterion)
    train_step = nn.TrainOneStepCell(net_with_loss, optimizer)

    for e in range(epochs):
        total_loss = 0.0
        # 模拟训练数据
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            img = input_data[start_idx:end_idx]
            loss = train_step(img, img)
            total_loss += loss.asnumpy()

            if i % 10 == 0 and i != 0:
                print(f'Epoch {e + 1}, Loss on batch {i}, {total_loss / 10:.5f}')
                print('-' * 10)
                total_loss = 0.0


if __name__ == '__main__':
    main()
