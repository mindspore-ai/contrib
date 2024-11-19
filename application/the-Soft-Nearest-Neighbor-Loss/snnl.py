import mindspore
from mindspore import nn, Tensor, context, ops
import numpy as np

class SNNLCrossEntropy(nn.Cell):
    STABILITY_EPS = 0.00001

    def __init__(self,
                 model,
                 temperature=100.0,
                 factor=-10.0,
                 optimize_temperature=True,
                 cos_distance=True,
                 layer_names=None):
        super(SNNLCrossEntropy, self).__init__()

        self.temperature = Tensor(temperature, mindspore.float32)
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
        self.model = model


    @staticmethod
    def pairwise_euclid_distance(A, B):
        """计算两个矩阵之间的成对欧氏距离。"""
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = ops.ReduceSum(keep_dims=True)(ops.Pow()(A, 2), 1).reshape((1, batchA))
        sqr_norm_B = ops.ReduceSum(keep_dims=True)(ops.Pow()(B, 2), 1).reshape((batchB, 1))
        inner_prod = ops.MatMul()(B, ops.Transpose()(A, (1, 0)))

        tile_1 = ops.Tile()(sqr_norm_A, (batchB, 1))
        tile_2 = ops.Tile()(sqr_norm_B, (1, batchA))
        return tile_1 + tile_2 - 2 * inner_prod

    @staticmethod
    def pairwise_cos_distance(A, B):
        """计算两个矩阵之间的成对余弦距离。"""
        normalized_A = ops.L2Normalize(axis=1, epsilon=1e-12)(A)
        normalized_B = ops.L2Normalize(axis=1, epsilon=1e-12)(B)
        prod = ops.MatMul()(normalized_A, ops.Transpose()(normalized_B, (1, 0)))
        return 1 - prod

    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
        return ops.Exp()(-(distance_matrix / temp))

    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """计算所有元素之间的行归一化指数化成对距离，作为每个元素选择邻居点的概率。"""
        eye_matrix = ops.eye(x.shape[0], x.shape[0], dtype=x.dtype)
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - eye_matrix
        summed_f = ops.ReduceSum(keep_dims=True)(f, 1)
        return f / (SNNLCrossEntropy.STABILITY_EPS + summed_f)

    @staticmethod
    def same_label_mask(y, y2):
        """生成掩码矩阵，当且仅当 y[i] == y2[j] 时，mask[i,j] = 1。"""
        return (ops.ExpandDims()(y, 1) == ops.ExpandDims()(y2, 0)).astype(mindspore.float32)

    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """计算共享标签的邻居点的成对采样概率。"""
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
            SNNLCrossEntropy.same_label_mask(y, y)

    @staticmethod
    def SNNL(x, y, temp=100.0, cos_distance=True):
        """软最近邻损失（Soft Nearest Neighbor Loss）。"""
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -ops.Log()(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()

    def construct(self, x, y):
        return SNNLCrossEntropy.SNNL(x, y, self.temperature, self.cos_distance)

if __name__ == '__main__':
    class SimpleNetMS(nn.Cell):
        def __init__(self, input_size=10, hidden_size=50, num_classes=3):
            super(SimpleNetMS, self).__init__()
            self.fc1 = nn.Dense(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Dense(hidden_size, num_classes)

        def construct(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    model_ms = SimpleNetMS(input_size=10, hidden_size=50, num_classes=3)

    np.random.seed(2)
    input_data = Tensor(np.random.randn(5, 10).astype(np.float32))
    labels = Tensor(np.random.randint(0, 3, size=(5,)).astype(np.int32))

    loss_fn = SNNLCrossEntropy(model=model_ms, temperature=100.0, cos_distance=True)

    features = model_ms(input_data)
    loss = loss_fn(features, labels)
    print("MindSpore Loss:", loss.asnumpy())