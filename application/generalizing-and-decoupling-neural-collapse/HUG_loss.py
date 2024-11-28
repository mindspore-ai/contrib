import mindspore as ms
from mindspore import ops, nn, Parameter, Tensor
import math
import numpy as np

class HUG_MHE(nn.Cell):
    def __init__(self, alpha=1., beta=1.):
        super(HUG_MHE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6  

    def construct(self, feat, classifier, target):
        feat = feat
        weight = classifier
        weight = weight / ops.norm(weight, 2, dim=1, keepdim=True)
        
        inner_pro = ops.cdist(weight, weight)
        inner_pro = ops.triu(inner_pro, diagonal=1)
        pro_mask = inner_pro > 0
        weight_wise = ops.mean(1. / (inner_pro[pro_mask] * inner_pro[pro_mask] + self.epsilon))
        
        sample_wise_all = 0
        class_indices = ops.unique(target)[0]
        for class_i in class_indices:
            mask = target == class_i
            feat_class = feat[mask]
            feat_class = feat_class / ops.norm(feat_class, 2, dim=1, keepdim=True)
            class_mean = weight[class_i].unsqueeze(0)
            concen_loss = ops.cdist(feat_class, class_mean)
            sample_wise = ops.mean(concen_loss)
            sample_wise_all += sample_wise
        return sample_wise_all / len(class_indices) * self.beta, weight_wise * self.alpha, feat

if __name__ == '__main__':
    # utilizing the random data to simulate the unconstrained feature
    # simulate CIFAR-100
    num_c = 100
    sample_c = 500 

    shape = (50000, 512)
    stdv = 1. / math.sqrt(shape[1])
    minval = -stdv
    maxval = stdv
    all_features_data = ops.uniform(shape, Tensor(minval, ms.float32), Tensor(maxval, ms.float32), dtype=ms.float32)
    all_features = Parameter(all_features_data)
    all_features.requires_grad = True

    shape = (num_c, 512)
    stdv = 1. / math.sqrt(shape[1])
    minval = -stdv
    maxval = stdv
    all_classifier_data = ops.uniform(shape, Tensor(minval, ms.float32), Tensor(maxval, ms.float32), dtype=ms.float32)
    all_classifier = Parameter(all_classifier_data)
    all_classifier.requires_grad = True

    all_labels = None

    for class_i in range(num_c):
        class_tensor = class_i * ops.ones((sample_c, 1), dtype=ms.int32)
        if all_labels is None:
            all_labels = class_tensor
        else:
            all_labels = ops.cat((all_labels, class_tensor), 0)

    optimizer = nn.SGD([{'params': [all_features]}, {'params': [all_classifier]}], learning_rate=1.0,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    criterion = HUG_MHE()

    def forward_fn(feat, classifier, labels):
        loss1, loss2, _ = criterion(feat, classifier, labels)
        return loss1 + loss2

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(feat, classifier, labels):
        loss, grads = grad_fn(feat, classifier, labels)
        optimizer(grads)
        return loss

    for epoch in range(200):
        for j in range(100):
            feat = all_features
            labels = all_labels.squeeze().long()
            try:
                total_loss = train_step(feat, all_classifier, labels)
            except Exception as e:
                print()
        loss1, loss2, _ = criterion(feat, all_classifier, labels)
        print(loss1.item(), loss2.item())