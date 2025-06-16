import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal, Uniform

class DietNet(nn.Cell):
    def __init__(self, feature_matrix, num_classes, device=None, number_of_genes=20530,
                 number_of_gene_features=173, embedding_size=500):
        """
        Class for the DietNetworks in MindSpore
        :param feature_matrix: Precomputed matrix of gene features of size (n_genes, n_features)
        :param num_classes: Number of classes in the prediction class
        """
        super(DietNet, self).__init__()
        self.embedding_size = embedding_size
        self.number_of_gene_features = number_of_gene_features
        self.number_of_genes = number_of_genes
        
        self._feature_matrix = ms.Parameter(feature_matrix, name="_feature_matrix", requires_grad=False)
        
        # 预测器网络
        self.predictor = nn.Dense(
            self.embedding_size, 
            num_classes,
            weight_init=Normal(0.02),
            bias_init='zeros'
        )

        # 辅助网络1 - 用于预测编码器权重
        self.aux1_layer1 = nn.Dense(
            self.number_of_gene_features, 
            self.embedding_size,
            weight_init=Normal(0.02),
            bias_init='zeros'
        )
        self.aux1_layer2 = nn.Dense(
            self.embedding_size, 
            self.embedding_size,
            weight_init=Normal(0.02),
            bias_init='zeros'
        )

        # 辅助网络2 - 用于预测解码器权重
        self.aux2_layer1 = nn.Dense(
            self.number_of_gene_features, 
            self.embedding_size,
            weight_init=Normal(0.02),
            bias_init='zeros'
        )
        self.aux2_layer2 = nn.Dense(
            self.embedding_size, 
            self.embedding_size,
            weight_init=Normal(0.02),
            bias_init='zeros'
        )
        
        self.relu = ops.ReLU()
        self.matmul = ops.MatMul()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """
        前向传播
        """
        feature_matrix_T = self.transpose(self._feature_matrix, (1, 0))  # 相当于 self.feature_matrix.T
        
        # 第一步: 计算编码器权重 W_e
        # 原式: W_e = self.aux1_layer2(F.relu(self.aux1_layer1(self.feature_matrix.T).T).T)
        aux1_step1 = self.aux1_layer1(feature_matrix_T)  # 对应 self.aux1_layer1(self.feature_matrix.T)
        aux1_step1_T = self.transpose(aux1_step1, (1, 0))  # 对应 .T
        aux1_step2 = self.relu(aux1_step1_T)  # 对应 F.relu()
        aux1_step2_T = self.transpose(aux1_step2, (1, 0))  # 对应 .T
        W_e = self.aux1_layer2(aux1_step2_T)  # 对应 self.aux1_layer2()
        
        # 计算潜在表示
        latent = self.relu(self.matmul(x, W_e))  # 对应 F.relu(torch.matmul(x, W_e))

        # 第二步: 计算解码器权重 W_d
        # 原式: W_d = self.aux2_layer2(F.relu(self.aux2_layer1(self.feature_matrix.T)))
        aux2_step1 = self.aux2_layer1(feature_matrix_T)  # 对应 self.aux2_layer1(self.feature_matrix.T)
        aux2_step2 = self.relu(aux2_step1)  # 对应 F.relu()
        W_d = self.aux2_layer2(aux2_step2)  # 对应 self.aux2_layer2()
        
        # 计算重构
        W_d_T = self.transpose(W_d, (1, 0))  # 对应 W_d.T
        x_hat = self.matmul(latent, W_d_T)  # 对应 torch.matmul(latent, W_d.T)

        # 计算预测
        y_hat = self.predictor(latent)  # 对应 self.predictor(latent)

        return x_hat, y_hat