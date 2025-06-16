# Selective_Backpropagation - MindSpore

This code is a mindspore implementation of Selective_Backpropagation 

## Requirements

```bash
# 核心依赖
mindspore>=2.0.0      # MindSpore深度学习框架
numpy>=1.19.0         # 数学计算库，用于数组操作
collections           # 内置库，用于创建deque

# 可选依赖
matplotlib>=3.3.0     # 用于可视化训练结果和测试性能
tqdm>=4.50.0          # 提供进度条显示
pandas>=1.1.0         # 用于数据处理和分析
scikit-learn>=0.24.0  # 可用于评估指标计算
pillow>=8.0.0         # 图像处理库，CIFAR-10数据可视化
urllib3>=1.26.0       # 用于下载CIFAR-10数据集
```

## Training Data

- 采用CIFAR-10数据集

#### 导入代码
```
def download_cifar10(target_dir="./data_ms"):
    """下载CIFAR-10数据集并解压到目标目录"""
    import os
    import urllib.request
    import tarfile
    
    print("正在下载CIFAR-10数据集...")
    
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # 定义下载URL和目标文件路径
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    filename = os.path.join(target_dir, "cifar-10-binary.tar.gz")
    
    # 下载文件
    urllib.request.urlretrieve(url, filename)
    print(f"下载完成: {filename}")
    
    # 解压文件
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print(f"解压完成: {os.path.join(target_dir, 'cifar-10-batches-bin')}")
    
    return True

```

## Reference
[paper link] https://arxiv.org/abs/1910.00762v1

[github link] https://github.com/Manuscrit/SelectiveBackPropagation