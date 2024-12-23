import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import numpy as np


class GE2ELoss(nn.Cell):
    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        '''
        实现 Generalized End-to-End (GE2E) 损失函数。

        参数:
            - init_w (float): 损失函数中缩放参数 w 的初始值。
            - init_b (float): 损失函数中偏置参数 b 的初始值。
            - loss_method (str): 损失计算方法，支持 'softmax' 或 'contrast'。
        '''
        super(GE2ELoss, self).__init__()
        self.w = Parameter(Tensor([init_w], dtype=mindspore.float32))
        self.b = Parameter(Tensor([init_b], dtype=mindspore.float32))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast'], "loss_method 必须为 'softmax' 或 'contrast'"

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        elif self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast
        self.mean = ops.ReduceMean(keep_dims=False)
        self.matmul = ops.MatMul(transpose_b=True)
        self.square = ops.Square()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.sqrt = ops.Sqrt()
        self.clip = ops.clip_by_value
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.sigmoid = ops.Sigmoid()
        self.max = ops.ReduceMax(keep_dims=False)
        self.concat = ops.Concat(axis=0)
        self.stack = ops.Stack(axis=0)
        self.expand_dims = ops.ExpandDims()

    def l2_norm(self, x, axis, keep_dims):
        '''
        计算 L2 范数。
        '''
        return self.sqrt(self.reduce_sum(self.square(x), axis=axis))

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        计算新的质心，排除参考话语。
        '''
        speaker_dvecs = dvecs[spkr]  # (M, D)
        excl_before = speaker_dvecs[:utt]  # (utt, D)
        excl_after = speaker_dvecs[utt + 1:]  # (M - utt -1, D)

        excl = self.concat((excl_before, excl_after))  # (M-1, D)
        new_centroid = self.mean(excl, axis=0)  # (D,)
        new_centroids = centroids.__deepcopy__(memodict=new_centroid)
        new_centroids[spkr] = new_centroid
        return new_centroids  # (N, D)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        计算余弦相似度矩阵，维度为 (N, M, N)。
        '''
        N, M, D = dvecs.shape
        cos_sim_matrix = []

        for spkr_idx in range(N):
            cs_row = []
            for utt_idx in range(M):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)  # (N, D)
                utterance = dvecs[spkr_idx, utt_idx].reshape(1, -1)  # (1, D)
                sim = self.matmul(utterance, new_centroids)  # (1, N)
                norm_utterance = self.l2_norm(utterance, axis=1, keep_dims=True)  # (1, 1)
                norm_centroids = self.l2_norm(new_centroids, axis=1, keep_dims=True)  # (N, 1)
                cos_sim = sim / (norm_centroids.reshape(1, N) * norm_utterance + 1e-6)
                cos_sim = self.clip(cos_sim, 1e-6, 1e6)
                cs_row.append(cos_sim)
            cs_row = ops.Concat(axis=1)(cs_row)  # (1, M, N)
            cos_sim_matrix.append(cs_row)
        cos_sim_matrix = self.concat(cos_sim_matrix)  # (N, M, N)
        return cos_sim_matrix

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        使用 softmax 计算每个嵌入的损失 L(e_{ji})。
        '''
        if len(cos_sim_matrix.shape) == 2:
            cos_sim_matrix = cos_sim_matrix.unsqueeze(2)  # 变为 (N, M, 1)
        N, M, C = cos_sim_matrix.shape
        L = []
        for j in range(N):
            cos_sim = cos_sim_matrix[j]  # (M, C)
            log_softmax = self.log_softmax(cos_sim)  # (M, C)
            if j < log_softmax.shape[1]:
                loss = -log_softmax[:, j]  # (M,)
            else:
                loss = -log_softmax[:, 0]  # 防止越界
            L.append(loss)
        L = self.stack(L)  # (N, M)
        return L

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        '''
        使用对比损失计算每个嵌入的损失 L(e_{ji})。
        '''
        N, M, C = cos_sim_matrix.shape
        L = []
        for j in range(N):
            loss_row = []
            for i in range(M):
                centroids_sigmoids = self.sigmoid(cos_sim_matrix[j, i])  # (C,)
                excl_centroids_sigmoids = ops.Concat(axis=0)(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1:]))  # (C-1,)
                max_excl = self.max(excl_centroids_sigmoids)  # scalar
                loss = 1.0 - centroids_sigmoids[j] + max_excl  # scalar
                loss_row.append(loss)
            loss_row = self.stack(loss_row)  # (M,)
            L.append(loss_row)
        L = self.stack(L)  # (N, M)
        return L

    def construct(self, dvecs):
        '''
        计算 GE2E 损失，输入维度为 (N, M, D)。
        '''
        centroids = self.mean(dvecs, axis=1)  # (N, D)
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)  # (N, M, N)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b  # (N, M, N)
        L = self.embed_loss(dvecs, cos_sim_matrix)  # (N, M)
        return ops.ReduceSum()(L)  # scalar


class SimpleNet(nn.Cell):
    def __init__(self, input_dim=128, embed_dim=256, loss_method='softmax'):
        super(SimpleNet, self).__init__()
        self.dense = nn.Dense(input_dim, embed_dim)
        self.relu = nn.ReLU()
        self.loss_fn = GE2ELoss(loss_method=loss_method)

    def construct(self, x):
        embeddings = self.relu(self.dense(x))
        loss = self.loss_fn(embeddings)
        return loss


if __name__ == "__main__":
    dvecs = Tensor(np.random.randn(4, 5, 128), dtype=mindspore.float32)
    net = SimpleNet(input_dim=128, embed_dim=256, loss_method='softmax')
    loss = net(dvecs)
    print("最终损失值:", loss.asnumpy())
