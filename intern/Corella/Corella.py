import numpy as np
import mindspore.dataset as ds
from mindspore.dataset.vision import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindvision.engine.callback import LossMonitor
from mindspore.train import Model
import mindspore.dataset.transforms as C

# Set the device context
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# input
q = input('Default(d) or Custom(c) input? ')
if q == 'c':
    DN, NS, Sigma, W = input('Input? ').split()
else:
    DN, NS, Sigma, W = 'MNIST', '[Iden,Iden]', '70', '[[1.0,-1.0]]'

# type fixing
if not (DN == 'MNIST' or DN == 'Fashion-MNIST' or DN == 'Cifar-10'):
    print('DatasetName is false; MNIST selected as the default.')
    DN = 'MNIST'
NS = list(map(str, NS[1:-1].strip().split(',')))[:2]
Sigma = float(Sigma)
exec('W = ' + W)
W = np.array(W, dtype=float)
N = len(W[0, :])
T = len(W[:, 0])
IW = np.ones((N, T + 1), dtype=float)
IW[:, 1:] = W.transpose()

print(DN, NS, Sigma)
print('N = \n {}'.format(N))
print('T = \n {}'.format(T))
print('[1,W^T] = \n {}:'.format(IW))

if DN == 'MNIST':
    # ------------ MNIST ------------ #
    meanI = [0.1307]
    stdI = [0.3040]

    transform_train = [
        ToTensor(),
        Normalize(mean=meanI, std=stdI),
    ]

    transform_test = [
        ToTensor(),
        Normalize(mean=meanI, std=stdI),
    ]

    # 下载数据集
    trainset = ds.MnistDataset(dataset_dir='./data' ,usage='train', shuffle=True)
    testset = ds.MnistDataset(dataset_dir='./data', usage='test', shuffle=False)

    # 定义转换操作，将label转换为int32
    label_transform = C.TypeCast(mindspore.int32)

    # 应用转换
    trainset = trainset.map(operations=transform_train, input_columns=["image"])
    trainset = trainset.map(operations=label_transform, input_columns=["label"])
    testset = testset.map(operations=transform_test, input_columns=["image"])
    testset = testset.map(operations=label_transform, input_columns=["label"])

    # 创建数据加载器
    trainloader = trainset.batch(batch_size=1024, num_parallel_workers=2)
    testloader = testset.batch(batch_size=1024, num_parallel_workers=2)

elif DN == 'Fashion-MNIST':
    # ------------ Fashion-MNIST ------------ #
    meanI = [0.2860]
    stdI = [0.3205]

    transform_train = [
        ToTensor(),
        Normalize(mean=meanI, std=stdI),
    ]

    transform_test = [
        ToTensor(),
        Normalize(mean=meanI, std=stdI),
    ]

    # 下载数据集
    trainset = ds.FashionMnistDataset(dataset_dir='./data', usage='train', shuffle=True)
    testset = ds.FashionMnistDataset(dataset_dir='./data', usage='test', shuffle=False)

    # 定义转换操作，将label转换为int32
    label_transform = C.TypeCast(mindspore.int32)

    # 应用转换
    trainset = trainset.map(operations=transform_train, input_columns=["image"])
    trainset = trainset.map(operations=label_transform, input_columns=["label"])
    testset = testset.map(operations=transform_test, input_columns=["image"])
    testset = testset.map(operations=label_transform, input_columns=["label"])

    # 创建数据加载器
    trainloader = trainset.batch(batch_size=1024, num_parallel_workers=2)
    testloader = testset.batch(batch_size=1024, num_parallel_workers=2)


elif DN == 'Cifar-10':
    # ------------ Cifar-10 ------------ #
    meanI = [0.4914, 0.4822, 0.4465]
    stdI = [0.2023, 0.1994, 0.2010]

    transform_train = [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=meanI, std=stdI),
    ]

    transform_test = [
        ToTensor(),
        Normalize(mean=meanI, std=stdI),
    ]

    trainset = ds.Cifar10Dataset(dataset_dir='./data', usage='train', shuffle=True)
    testset = ds.Cifar10Dataset(dataset_dir='./data', usage='test', shuffle=False)

    # 定义转换操作，将label转换为int32
    label_transform = C.TypeCast(mindspore.int32)

    # 应用转换
    trainset = trainset.map(operations=transform_train, input_columns=["image"])
    trainset = trainset.map(operations=label_transform, input_columns=["label"])
    testset = testset.map(operations=transform_test, input_columns=["image"])
    testset = testset.map(operations=label_transform, input_columns=["label"])

    # 创建数据加载器
    trainloader = trainset.batch(batch_size=1024, num_parallel_workers=2)
    testloader = testset.batch(batch_size=1024, num_parallel_workers=2)

isRGB = int(DN == 'Cifar-10')  # MNIST and Fashion-MNIST: 0   # Cifar-10: 1

if NS[1] == 'Iden':
    NS1 = 10
else:
    NS1 = int(NS[1])

# ---------------- \bar{g}_0 ---------------- #
if NS[0] == 'Iden':
    class g_0(nn.Cell):
        def __init__(self):
            super(g_0, self).__init__()

        def construct(self, X):
            out = X
            return out

else:
    class g_0(nn.Cell):
        def __init__(self):
            super(g_0, self).__init__()
            self.client_L1_pad = nn.ConstantPad2d((1 - isRGB, 0, 1 - isRGB, 0), 0)
            self.client_L2_conv2d = nn.Conv2d(1 + 2 * isRGB, int(NS[0]), kernel_size=5, stride=3, pad_mode="valid",
                                              has_bias=True)

        def construct(self, X):
            out = self.client_L1_pad(X)
            out = self.client_L2_conv2d(out)
            out = ops.relu(out)
            return out

# ---------------- f_j ---------------- #
if NS[0] == 'Iden':
    class f_j(nn.Cell):
        def __init__(self):
            super(f_j, self).__init__()
            self.server_L1_pad = nn.ConstantPad2d((1 - isRGB, 0, 1 - isRGB, 0), 0)
            self.server_L2_conv2d = nn.Conv2d(1 + 2 * isRGB, 64, kernel_size=5, stride=3, pad_mode='valid',
                                              has_bias=True)
            self.server_L3_conv2d = nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode='valid', has_bias=True)
            self.server_L4_fc = nn.Dense(128 * (7 + isRGB) * (7 + isRGB), 1024)
            self.server_L5_fc = nn.Dense(1024, NS1)

        def construct(self, X):
            out = self.server_L1_pad(X)
            out = self.server_L2_conv2d(out)
            out = ops.relu(out)
            out = self.server_L3_conv2d(out)
            out = ops.relu(out)
            out = out.reshape(out.shape[0], -1)
            out = self.server_L4_fc(out)
            out = ops.relu(out)
            out = self.server_L5_fc(out)
            return out

else:
    class f_j(nn.Cell):
        def __init__(self):
            super(f_j, self).__init__()
            self.server_L1_conv2d = nn.Conv2d(int(NS[0]), 128, kernel_size=3, stride=1, pad_mode='valid', has_bias=True)
            self.server_L2_fc = nn.Dense(128 * (7 + isRGB) * (7 + isRGB), 1024)
            self.server_L3_fc = nn.Dense(1024, NS1)

        def construct(self, X):
            out = self.server_L1_conv2d(X)
            out = ops.relu(out)
            out = out.reshape(out.shape[0], -1)
            out = self.server_L2_fc(out)
            out = ops.relu(out)
            out = self.server_L3_fc(out)
            return out

# ---------------- \bar{g}_1 ---------------- #
if NS[1] == 'Iden':
    class g_1(nn.Cell):
        def __init__(self):
            super(g_1, self).__init__()

        def construct(self, X):
            out = X
            return out
else:
    class g_1(nn.Cell):
        def __init__(self):
            super(g_1, self).__init__()
            self.client_L2_fc = nn.Dense(NS1, 10)

        def construct(self, X):
            out = ops.relu(X)
            out = self.client_L2_fc(out)
            return out


# ---------------- network ---------------- #
class network(nn.Cell):
    def __init__(self):
        super(network, self).__init__()
        # client
        self.client_g_0 = g_0()
        # servers
        for j in range(N):
            exec('self.server{} = f_j()'.format(j + 1))
        # client
        self.client_g_1 = g_1()

    def construct(self, X):
        Z_scale = ZR
        U = self.client_g_0(X)
        # normalize
        mean_U = ops.mean(U, [1, 2, 3], keep_dims=True)
        U = U - mean_U * ops.ones(U.shape)  # zero mean
        variance_U = ops.mean((U - mean_U) ** 2, [1, 2, 3], keep_dims=True)
        U = U / ops.sqrt(variance_U) * ops.ones(U.shape)  # set variance to 1
        # add noise (queries)
        Q = []
        Z = (Z_scale * ops.randn(list(U.shape) + [T]))  # primary noise
        for j in range(N):
            q = U * IW[j, 0]
            for t in range(T):
                q += Z[:, :, :, :, t] * IW[j, t + 1]
            normP = ops.sqrt(mindspore.Tensor(IW[j, 0] ** 2 + sum((Z_scale * IW[j, 1:]) ** 2),
                                              mindspore.float32))  # for normalizing the queries
            Q += [q / normP]
        # answers
        A = []
        for j in range(N):
            exec('A += [self.server{}(Q[{}])]'.format(j + 1, j))
        # sumA
        sumA = A[0]
        for j in range(1, N):
            sumA += A[j]
        # out
        out = self.client_g_1(sumA)
        return out


net = network()
TotalEpochs = int(np.ceil((-1 + np.sqrt(1 + 8 / 0.002 * Sigma)) / 2))  # solving [0.002 * (n * (n + 1) / 2) = Sigma]
step_size = 0
ZR = 0
net_loss = nn.CrossEntropyLoss()

optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)
model = Model(net, loss_fn=net_loss, optimizer=optimizer, metrics={'accuracy'})

for epoch_num in range(TotalEpochs):
    step_size += 0.002
    ZR += step_size
    model.train(1, trainloader, callbacks=[LossMonitor(0.01, 1875)])
    acc = model.eval(testloader)
    print(f"epoch {epoch_num} : {acc}")
