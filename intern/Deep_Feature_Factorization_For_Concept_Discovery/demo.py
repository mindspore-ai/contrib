#使用imageNET 1k数据集，加载本地模型文件https://www.mindspore.cn/resources/hub/details?MindSpore/1.9/vgg19_imagenet2012

import os
import numpy as np
from matplotlib import pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindspore.nn import Cell
from nmf import NMF
from utils import imresize, show_heatmaps

# 设置运行环境为 CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 数据路径
data_path = 'E:/Code/Dataset/imageNet/val/n01440764'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"数据路径 {data_path} 不存在，请检查路径或提供有效数据目录")

filenames = os.listdir(data_path)
max_images = 6
filenames = filenames[:max_images]

# 加载并预处理图像
raw_images = [plt.imread(os.path.join(data_path, filename)) for filename in filenames]
raw_images = [imresize(img, 224, 224) for img in raw_images]
raw_images = np.stack(raw_images)

print(f"原始图像形状: {raw_images.shape}")
print(f"原始图像数值范围: min={raw_images.min():.4f}, max={raw_images.max():.4f}")

images = raw_images.transpose((0, 3, 1, 2)).astype('float32')
images -= np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
images /= np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

print(f"预处理后图像形状: {images.shape}")
print(f"预处理后图像数值范围: min={images.min():.4f}, max={images.max():.4f}")

images = Tensor(images, dtype=ms.float32)


# 定义 VGG19 网络结构
class VGG19(Cell):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def construct(self, x):
        return self.features(x)


# 加载预训练模型并调整参数名
net = VGG19()
param_dict = load_checkpoint('vgg19.ckpt')
new_param_dict = {}
for old_name, param in param_dict.items():
    if old_name.startswith('layers'):  # 只处理特征提取层的参数
        new_name = old_name.replace('layers', 'features')  # 将 'layers' 替换为 'features'
        new_param_dict[new_name] = param
load_param_into_net(net, new_param_dict)

features = net.features[:-1]
features_output = ops.stop_gradient(features(images))

# 调试特征输出
features_output_np = features_output.asnumpy()
print(f"特征输出形状: {features_output.shape}")
print(f"特征输出数值范围: min={features_output_np.min():.4f}, max={features_output_np.max():.4f}")
print(f"特征输出是否存在NaN: {np.isnan(features_output_np).any()}")
print(f"特征输出是否全为0: {(features_output_np == 0).all()}")
print(f"特征输出样本（前10个值）: {features_output_np.flatten()[:10]}")

flat_features = ops.transpose(features_output, (0, 2, 3, 1)).reshape((-1, features_output.shape[1]))
print(f'重塑特征从 {features_output.shape} 到 {flat_features.shape}')
flat_features_np = flat_features.asnumpy()
print(f"重塑后特征数值范围: min={flat_features_np.min():.4f}, max={flat_features_np.max():.4f}")

plt.ion()
print("显示原始图像...")
show_heatmaps(raw_images, None, 0, enhance=1)
plt.pause(1)

for K in range(1, 5):
    print(f"处理 K={K}...")
    W, _ = NMF(flat_features, K, random_seed=50)

    W_np = W.asnumpy()
    print(f"K={K} 时 W 形状: {W.shape}")
    print(f"W 数值范围: min={W_np.min():.4f}, max={W_np.max():.4f}")
    print(f"W 是否全为0: {(W_np == 0).all()}")

    heatmaps = W.reshape(features_output.shape[0], features_output.shape[2], features_output.shape[3], K)
    heatmaps = ops.transpose(heatmaps, (0, 3, 1, 2))
    heatmaps = ops.interpolate(heatmaps, size=(224, 224), mode='bilinear', align_corners=True)
    heatmaps_np = heatmaps.asnumpy()

    print(f"热图形状: {heatmaps.shape}")
    print(f"热图数值范围: min={heatmaps_np.min():.4f}, max={heatmaps_np.max():.4f}")

    show_heatmaps(raw_images, heatmaps_np, K, enhance=1)
    plt.pause(1)

plt.ioff()
print("所有图像处理完成，请关闭窗口以结束程序")
plt.show()

print("程序结束")