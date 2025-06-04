import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from sklearn.model_selection import train_test_split

from dietnetworks_ms import DietNet
from training_ms import run_training_diet

np.random.seed(42)
ms.set_seed(42)
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU") 

def generate_simulated_data(num_samples=300, num_genes=1000, num_classes=2):    
    # 生成基因表达数据 (均值为0，方差为1的正态分布)
    X = np.random.randn(num_samples, num_genes)
    y = np.zeros(num_samples, dtype=np.int32)
    samples_per_class = num_samples // num_classes
    
    for i in range(num_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class if i < num_classes - 1 else num_samples
        
        shift = np.random.randn(num_genes) * 0.1  # 随机偏移
        X[start_idx:end_idx, :] += shift
        y[start_idx:end_idx] = i
    
    return X, y

def generate_feature_matrix(num_genes=1000, feature_dim=173):
    feature_matrix = np.random.randn(num_genes, feature_dim) * 0.1
    return feature_matrix


class GeneExpressionDataset:
    """MindSpore自定义数据集类"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)


def create_dataset(features, labels, batch_size=32, shuffle=True, num_parallel_workers=1):
    # 创建数据集实例
    dataset_generator = GeneExpressionDataset(features, labels)
    
    # 转换为MindSpore数据集
    dataset = ds.GeneratorDataset(
        dataset_generator, 
        column_names=["features", "labels"],
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)  
    return dataset

def main():
    num_samples = 300      
    num_genes = 1000       
    num_classes = 2        
    feature_dim = 173      
    batch_size = 32
    learning_rate = 0.001  
    num_epochs = 50
    embedding_size = 128   
    test_proportion = 0.3  
    
    X, y = generate_simulated_data(num_samples=num_samples, num_genes=num_genes, num_classes=num_classes)
    
    feature_matrix = generate_feature_matrix(num_genes=num_genes, feature_dim=feature_dim)
    feature_matrix = feature_matrix.T

    train_samples = int(num_samples * (1 - test_proportion))
    test_samples = num_samples - train_samples
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_samples,
        test_size=test_samples,
        stratify=y,
        random_state=0,
        shuffle=True
    )
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)   
    train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_dataset = create_dataset(X_test, y_test, batch_size=len(X_test), shuffle=False)
    
    feature_matrix = ms.Tensor(feature_matrix, dtype=ms.float32)
    
    model = DietNet(
        feature_matrix=feature_matrix,
        num_classes=num_classes,
        number_of_genes=num_genes,
        number_of_gene_features=feature_dim,
        embedding_size=embedding_size
    )
    
    history = run_training_diet(
        model=model,
        nb_epochs=num_epochs,
        train_loader=train_dataset,
        test_loader=test_dataset,
        lr=learning_rate
    )
    
    print("done")

if __name__ == "__main__":
    main()