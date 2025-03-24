import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from mindcv.models import inception_v3
import numpy as np
from scipy.stats import entropy


def inception_score(imgs, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # 自动选择运行设备
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="GPU" if ms.context.get_context("device_target") == "GPU" else "CPU")

    # 创建数据加载器
    dataloader = imgs.batch(batch_size, drop_remainder=False)

    # 加载预训练模型
    inception_model = inception_v3(pretrained=True)
    inception_model.set_train(False)  # 推理模式

    def get_pred(x):
        if resize:
            x = ops.ResizeBilinearV2()(x, (299, 299)) # 调整图像大小,如果需要resize
        x = ops.Cast()(x, ms.float32)
        x = inception_model(x)
        return ops.Softmax(axis=1)(x).asnumpy()

    preds = np.zeros((N, 1000))

    # 批量推理
    idx = 0 # 模拟索引
    for batch in dataloader.create_tuple_iterator():
        # print(idx / 32) # 查看推理进度
        batch = batch[0]
        batch_preds = get_pred(batch)
        batch_size = batch_preds.shape[0]
        preds[idx:idx + batch_size] = batch_preds
        idx += batch_size

    # 计算平均KL散度
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits)]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class IgnoreLabelDataset:
    def __init__(self, orig):
        self.data = []
        for item in orig.create_dict_iterator():
            img = item['image'].asnumpy()
            self.data.append(img)

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from mindspore.dataset import Cifar10Dataset
    import mindspore.dataset.vision as vision
    from mindspore.dataset.transforms import Compose

    transform = Compose([
        vision.Resize((32, 32)),
        vision.ToTensor(),
        vision.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准归一化
                         std=[0.229, 0.224, 0.225],
                         is_hwc=False)
    ])

    # 加载数据集
    cifar = Cifar10Dataset(dataset_dir='./data', usage='train', shuffle=True)
    cifar = cifar.map(operations=transform, input_columns="image")

    # 创建数据加载器
    dataset = IgnoreLabelDataset(cifar)
    dataloader = GeneratorDataset(dataset, column_names=["image"], shuffle=False)

    # 计算Inception Score
    print("Calculating Inception Score...")
    print(inception_score(dataloader, batch_size=32, resize=True, splits=10))