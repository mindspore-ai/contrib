import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Normal, initializer

mean = 1e-8
std = 1e-9
encoder_features = 512
total_epochs = 150


class NCODLoss(nn.Cell):
    def __init__(self, sample_labels, num_examp=50000, num_classes=100, ratio_consistency=0, ratio_balance=0):
        super(NCODLoss, self).__init__()

        self.num_classes = num_classes
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        self.u = Parameter(initializer(Normal(sigma=std, mean=mean), [num_examp, 1], mindspore.float32), name='u')

        self.beginning = True
        self.prevSimilarity = Parameter(Tensor(np.random.rand(num_examp, encoder_features).astype(np.float32)),
                                        requires_grad=False, name='prevSimilarity')
        self.masterVector = Parameter(Tensor(np.random.rand(num_classes, encoder_features).astype(np.float32)),
                                      requires_grad=False, name='masterVector')
        self.sample_labels = sample_labels
        self.bins = []

        for i in range(num_classes):
            self.bins.append(np.where(self.sample_labels == i)[0])

        self.split = ops.Split(axis=0, output_num=2)
        self.gather = ops.Gather()
        self.topk = ops.TopK()
        self.squeeze = ops.Squeeze(1)
        self.transpose = ops.Transpose()
        self.softmax = nn.Softmax(axis=1)
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.log = ops.Log()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sum = ops.ReduceSum(keep_dims=False)
        self.argmax = ops.Argmax(axis=1)
        self.one_hot = ops.OneHot()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.expand_dims = ops.ExpandDims()
        self.clip_by_value = ops.clip_by_value
        self.less = ops.Less()
        self.select = ops.Select()
        self.tensor_scatter_update = ops.TensorScatterUpdate()
        self.square = ops.Square()
        self.matmul = ops.MatMul()

    def construct(self, index, outputs, label, out, flag, epoch):
        if outputs.shape[0] > index.shape[0]:
            output, output2 = self.split(outputs)
            out1, out2 = self.split(out)
        else:
            output = outputs
            out1 = out

        eps = 1e-4

        u = self.gather(self.u, index, 0)

        if flag == 0:
            if self.beginning:
                percent = math.ceil((50 - (50 / total_epochs) * epoch) + 50)
                for i in range(len(self.bins)):
                    bin_indices = Tensor(self.bins[i], mindspore.int32)
                    u_squeezed = self.squeeze(self.u)
                    class_u = self.gather(u_squeezed, bin_indices, 0)
                    bottomK = int((len(class_u) / 100) * percent)
                    neg_class_u = -class_u
                    values, indices = self.topk(neg_class_u, bottomK)
                    important_indices = self.reshape(indices, (-1,))
                    prev_similarity_bin = self.gather(self.prevSimilarity, bin_indices, 0)
                    important_prev_similarity = self.gather(prev_similarity_bin, important_indices, 0)
                    mean_vector = self.mean(important_prev_similarity, 0)
                    indices_i = Tensor([[i]], mindspore.int32)
                    updates = self.expand_dims(mean_vector, 0)
                    self.masterVector = self.tensor_scatter_update(self.masterVector, indices_i, updates)
                self.beginning = True

            masterVector_norm = ops.norm(self.masterVector, dim=1, keepdim=True)
            masterVector_normalized = self.masterVector / masterVector_norm
            self.masterVector_transpose = self.transpose(masterVector_normalized, (1, 0))

        indices = self.reshape(index, (-1, 1))
        self.prevSimilarity = self.tensor_scatter_update(self.prevSimilarity, indices, out1)

        prediction = self.softmax(output)

        out_norm = ops.norm(out1, dim=1, keepdim=True)
        out_normalized = out1 / out_norm

        similarity = self.matmul(out_normalized, self.masterVector_transpose)
        similarity = similarity * label
        sim_mask = self.cast(similarity > 0.000, mindspore.float32)
        similarity = similarity * sim_mask

        u = u * label

        prediction = self.clip_by_value(prediction + u, eps, 1.0)
        loss = self.mean(-self.sum(similarity * self.log(prediction), 1))

        label_one_hot = self.soft_to_hard(output)

        diff = (label_one_hot + u) - label
        diff_squared = self.square(diff)
        MSE_loss = self.sum(diff_squared) / len(label)
        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = self.mean(prediction, 0)
            prior_distr = Tensor(np.ones_like(avg_prediction.asnumpy()) * (1.0 / self.num_classes), mindspore.float32)
            avg_prediction = self.clip_by_value(avg_prediction, eps, 1.0)
            balance_kl = self.mean(-self.sum(prior_distr * self.log(avg_prediction), 0))
            loss += self.ratio_balance * balance_kl

        if (outputs.shape[0] > index.shape[0]) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(output, output2)
            loss += self.ratio_consistency * self.mean(consistency_loss)

        return loss

    def consistency_loss(self, output1, output2):
        preds1 = self.softmax(output1)
        preds2 = self.log_softmax(output2)
        kldiv = preds1 * (self.log(preds1) - preds2)
        loss_kldiv = self.sum(kldiv, 1)
        return loss_kldiv

    def soft_to_hard(self, x):
        indices = self.argmax(x)
        depth = self.num_classes
        one_hot = self.one_hot(indices, depth, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32))
        return one_hot


if __name__ == '__main__':
    sample_labels = np.random.randint(0, 100, size=50000)
    loss_func_ms = NCODLoss(sample_labels)

    index = Tensor(np.random.randint(0, 50000, size=(128,)), mindspore.int32)
    outputs = Tensor(np.random.randn(256, 100).astype(np.float32))
    label = Tensor(np.random.randn(128, 100).astype(np.float32))
    out = Tensor(np.random.randn(256, 512).astype(np.float32))
    flag = 0
    epoch = 10

    loss_ms = loss_func_ms(index, outputs, label, out, flag, epoch)
    print("Loss:", loss_ms.asnumpy())