import mindspore
from mindspore import nn, ops


class CircleLoss(nn.Cell):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity
        self.normalizer = ops.L2Normalize(1)

    def construct(self, feats, labels):
        
        assert feats.shape[0] == labels.shape[0], \
            f"feats.shape[0]: {feats.shape[0]} is not equal to labels.shape[0]: {labels.shape[0]}"
        
        m = labels.shape[0]
        mask = labels.broadcast_to((m, m)).t().eq(labels.broadcast_to((m, m))).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs().triu(diagonal=1)
        # print(neg_mask)
        if self.similarity == 'dot':
            sim_mat = ops.matmul(feats, ops.t(feats))
        elif self.similarity == 'cos':
            feats = self.normalizer(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = ops.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = ops.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = ops.sum(ops.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = ops.sum(ops.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = ops.log(1 + loss_p * loss_n)
        return loss


if __name__ == '__main__':
    batch_size = 10
    feats = ops.rand(batch_size, 1028)
    labels = ops.randint(low=0, high=10, dtype=mindspore.int32, size=(batch_size,))
    circleloss = CircleLoss(similarity='cos')
    print(circleloss(feats, labels))