#============= mimic_utils.py =============
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import mindspore as ms
from mindspore import nn, Tensor, dataset
from mindspore.dataset import GeneratorDataset
from typing import Tuple, Dict

class DataGenerator:
    """内存友好的数据生成器"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features.astype(np.float32)
        self.labels = labels.reshape(-1, 1).astype(np.float32)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[GeneratorDataset, GeneratorDataset]:
    """数据划分函数（兼容类别不平衡）"""
    # 转换为numpy数组
    X_array = X.values.astype(np.float32)
    y_array = y.values.astype(np.int32)
    
    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array,
        test_size=test_size,
        stratify=y_array,
        random_state=42
    )
    
    # 创建数据集
    train_ds = GeneratorDataset(
        source=DataGenerator(X_train, y_train),
        column_names=["data", "label"],
        shuffle=True
    )
    
    test_ds = GeneratorDataset(
        source=DataGenerator(X_test, y_test),
        column_names=["data", "label"],
        shuffle=False
    )
    
    return train_ds, test_ds

def create_data_pipeline(dataset: GeneratorDataset, batch_size: int = 32) -> dataset.BatchDataset:
    """创建数据管道"""
    return dataset.batch(batch_size, drop_remainder=False)

def evaluate_model(model: nn.Cell, dataset: dataset.BatchDataset) -> Dict[str, float]:
    """修正键名后的评估函数"""
    model.set_train(False)
    
    all_preds = []
    all_labels = []
    
    for batch in dataset:
        data, labels = batch
        preds = model(data)
        all_preds.append(preds.asnumpy())
        all_labels.append(labels.asnumpy())
    
    preds_array = np.concatenate(all_preds, axis=0).flatten()
    labels_array = np.concatenate(all_labels, axis=0).flatten()
    
    return {
        "accuracy": accuracy_score(labels_array, (preds_array > 0.5).astype(int)),
        "auc": roc_auc_score(labels_array, preds_array),
        "avg_prob": preds_array.mean()  # 修正键名
    }

def save_metrics(metrics: Dict, filename: str = "metrics.json"):
    """保存评估指标"""
    import json
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filename}")

#============= 依赖检查 =============
required_packages = [
    ("mindspore", "2.2.10"),    # 深度学习框架
    ("pandas", "2.0.3"),        # 数据处理
    ("numpy", "1.24.3"),        # 数值计算
    ("scikit-learn", "1.3.0"),  # 评估指标
    ("chardet", "5.1.0")        # 编码检测
]

def check_dependencies():
    """检查依赖包版本"""
    import importlib.metadata
    missing = []
    outdated = []
    
    for pkg, req_version in required_packages:
        try:
            installed = importlib.metadata.version(pkg)
            if installed < req_version:
                outdated.append(f"{pkg} (installed: {installed}, required: {req_version})")
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)
    
    if missing or outdated:
        print("依赖检查不通过：")
        if missing:
            print("\n缺失的包：")
            print('\n'.join(f"- {pkg}" for pkg in missing))
        if outdated:
            print("\n需要升级的包：")
            print('\n'.join(f"- {info}" for info in outdated))
        raise ImportError("缺少必要的依赖包")
    else:
        print("所有依赖包检查通过")

if __name__ == "__main__":
    # 执行依赖检查
    check_dependencies()
    
    # 测试数据生成
    X_demo = pd.DataFrame(np.random.rand(100, 10))
    y_demo = pd.Series(np.random.randint(0, 2, 100))
    
    train_ds, test_ds = split_data(X_demo, y_demo)
    print("\n数据集测试：")
    print(f"训练样本数: {train_ds.get_dataset_size()}")
    print(f"测试样本数: {test_ds.get_dataset_size()}")