# pro.py
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, dataset as ds, set_seed

# ====================== 1. 数据集定义 ======================
class PairDataset:
    """生成(X1, X2)对的数据集"""
    def __init__(self, X1, X2, c, max_size=None):
        self.X1 = X1.astype(np.float32)
        self.X2 = X2.astype(np.float32)
        self.c = np.full((max(X1.shape[0], X2.shape[0]),), c, dtype=np.float32)
        self.max_size = max_size or X1.shape[0] * X2.shape[0]

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        idx1 = np.random.randint(0, self.X1.shape[0])
        idx2 = np.random.randint(0, self.X2.shape[0])
        merged_input = np.concatenate([self.X1[idx1], self.X2[idx2]])
        return merged_input, self.c[idx1]

# ====================== 2. 网络结构定义 ======================
class FeatureEncoder(nn.Cell):
    def __init__(self, input_dim=50, hidden_dim=20):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(input_dim, hidden_dim),
            nn.ReLU()
        )
        
    def construct(self, x):
        return self.net(x)

class Classifier(nn.Cell):
    def __init__(self, input_dim=40):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(input_dim, 1)
        )
        
    def construct(self, x):
        return self.net(x)

class PROWrapper(nn.Cell):
    def __init__(self, feature_encoder, classifier):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.classifier = classifier
        self.split = ops.Split(axis=1, output_num=2)
        
    def construct(self, x):
        x1, x2 = self.split(x)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        z1 = self.feature_encoder(x1)
        z2 = self.feature_encoder(x2)
        return self.classifier(ops.cat([z1, z2], axis=1))

# ====================== 3. 主训练流程 ======================
def generate_mock_data(samples=1000, features=50):
    set_seed(42)
    X_pos = np.random.randn(samples//2, features).astype(np.float32)
    X_neg = np.random.randn(samples//2, features).astype(np.float32) + 2  
    X_test = np.concatenate([X_pos[:100], X_neg[:100]], axis=0)
    return X_pos, X_neg, X_test

def main():
    set_seed(42)
    ms.set_context(device_target="CPU")
    
    # 生成数据
    X_pos, X_neg, X_test = generate_mock_data()
    print(f"[数据信息] 正样本: {X_pos.shape}, 负样本: {X_neg.shape}, 测试集: {X_test.shape}")

    # 初始化模型
    encoder = FeatureEncoder()
    classifier = Classifier()
    net = PROWrapper(encoder, classifier)
    
    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.RMSProp(net.trainable_params(), 
                          learning_rate=0.001,
                          weight_decay=1e-2)
    
    # 构建训练流程
    model = ms.Model(net, loss_fn, optimizer)
    
    # 准备数据集
    train_dataset = ds.GeneratorDataset(
        source=PairDataset(X_pos, X_neg, c=4.0, max_size=10000),
        column_names=["data", "label"],
        shuffle=True,
        num_parallel_workers=1
    ).batch(512)
    
    # 执行训练
    print("开始训练...")
    model.train(epoch=10, 
               train_dataset=train_dataset,
               callbacks=[ms.train.LossMonitor(per_print_times=100)])
    
    # 预测逻辑（简化版）
    print("\n执行预测...")
    test_pairs = np.concatenate([X_test, X_neg[:200]], axis=1)
    test_input = Tensor(test_pairs.astype(np.float32))
    scores = net(test_input).asnumpy().squeeze()
    
    anomalies = (scores > 2.5).astype(int)
    print("前20个样本的检测结果:", anomalies[:20])

if __name__ == "__main__":
    main()