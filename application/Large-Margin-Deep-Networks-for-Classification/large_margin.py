import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.numpy as ms_np


class LargeMarginLoss(nn.LossBase):
    def __init__(self,
                 gamma=10.0,
                 alpha_factor=0.25,
                 top_k=1,
                 epsilon=1e-6):
        super(LargeMarginLoss, self).__init__()  # 显式调用父类初始化
        self.gamma = gamma
        self.alpha_factor = alpha_factor
        self.dist_upper = gamma
        self.dist_lower = gamma * (1.0 - alpha_factor)
        self.top_k = top_k
        self.eps = epsilon

        self.softmax = nn.Softmax(axis=1)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.expand_dims = ops.ExpandDims()
        self.square = ops.Square()
        self.log = ops.Log()
        self.clip = ops.clip_by_value  # Correct usage of clip

    def safe_norm(self, x, axis=(1, 2, 3)):
        squared = self.square(x)
        sum_squared = self.reduce_sum(squared, axis)  # 移除 keep_dims 参数
        return ops.Sqrt()(sum_squared + self.eps)

    def construct(self, logits, onehot_labels, feature_maps):

        logits = self.clip(logits, -20, 20)
        prob = self.softmax(logits)


        # 先执行 reduce_sum，然后手动扩展维度
        correct_prob = self.reduce_sum(prob * onehot_labels, 1)
        correct_prob = self.expand_dims(correct_prob, 1)

        other_prob = prob * (1.0 - onehot_labels)


        if self.top_k > 1:
            topk_prob, _ = ops.TopK(sorted=True)(other_prob, self.top_k)
        else:
            topk_prob, _ = ops.ArgMaxWithValue(axis=1, keep_dims=True)(other_prob)

        diff_prob = correct_prob - topk_prob

        feature_losses = []
        for feature_map in feature_maps:
            feature_norm = self.safe_norm(feature_map)
            dist = diff_prob / (feature_norm + self.eps)
            dist = self.clip(dist, -1 / self.eps, 1 / self.eps)
            loss_term = ms_np.maximum(dist, self.dist_lower)
            loss_term = ms_np.maximum(0, self.dist_upper - loss_term) - self.dist_upper
            feature_losses.append(loss_term)

        final_loss = self.reduce_mean(ops.concat(feature_losses))
        final_loss = final_loss + 1e-6 * self.reduce_mean(self.square(logits))
        return final_loss


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()  # 确保正确调用父类初始化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = ops.Flatten()
        self.fc1 = nn.Dense(64 * 7 * 7, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Correct the parameter name to 'p'
        self.fc2 = nn.Dense(256, 10)

    def construct(self, x):  # 确保construct方法正确定义
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        conv1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        conv2 = x

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits, [conv1, conv2]  # 确保返回两个值
