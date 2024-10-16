import mindspore
from mindspore import ops, Tensor, nn
import mindspore.numpy as np

def cross_sigmoid_focal_loss(inputs,
                             targets,
                             gt_avoid_classes=None,
                             alpha=0.25,
                             gamma=2,
                             ignore_label=-1,
                             reduction="sum"):
    """
    Arguments:
       - inputs: inputs Tensor (N * C)
       - targets: targets Tensor (N)
       - gt_avoid_classes: [set(), set()...] neg label need to be avoided for each class
       - alpha: focal loss alpha
       - gamma: focal loss gamma
       - ignore_label: ignore_label default = -1
       - reduction: default = sum
    """

    def get_classes_idx(gt_avoid_classes, neg_target):
        classes_idx = []
        for idx, item in enumerate(gt_avoid_classes):
            if neg_target in item:
                classes_idx.append(idx)
        return classes_idx

    assert gt_avoid_classes is not None, "gt_avoid_classes must be provided"
    
    sample_mask = (targets != ignore_label)
    if (~sample_mask).sum() > 0:
        inputs = inputs[sample_mask]
        targets = targets[sample_mask]

    cross_mask = ops.ones(inputs.shape, dtype=mindspore.int8)
    t_mask = ops.zeros(inputs.shape[0], dtype=mindspore.int8)

    neg_targets = set()
    for item in gt_avoid_classes:
        neg_targets = neg_targets.union(item)

    for neg_target in neg_targets:
        neg_mask = targets == neg_target
        neg_idx = ops.nonzero(neg_mask).reshape(-1)
        if len(neg_idx) > 0:
            t_mask |= neg_mask
            cls_neg_idx = get_classes_idx(gt_avoid_classes, neg_target)
            cross_mask[neg_idx, cls_neg_idx] = 0

    vaild_idx = ops.nonzero(1 - t_mask).reshape(-1)
    pos_num = max(vaild_idx.shape[0], 1)
    expand_label_targets = ops.zeros_like(inputs)
    expand_label_targets[vaild_idx, targets[vaild_idx] - 1] = 1

    sigmoid = nn.Sigmoid()
    p = sigmoid(inputs)

    bce_with_logits = ops.BinaryCrossEntropy(reduction='none')
    ce_loss = bce_with_logits(inputs, expand_label_targets)

    p_t = p * expand_label_targets + (1 - p) * (1 - expand_label_targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * expand_label_targets + (1 - alpha) * (1 - expand_label_targets)
        loss = alpha_t * loss

    loss *= cross_mask

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
        loss /= pos_num
    return loss


if __name__ == '__main__':
    # 示例输入
    inputs = mindspore.ops.rand(10, 2)
    print(inputs)
    targets = mindspore.Tensor([1, 1, 1, 3, 3, 3, 4, 4, 4, 2]).long()
    gt_avoid_classes = [{4}, {3}]

    # 计算损失
    loss = cross_sigmoid_focal_loss(inputs, targets, gt_avoid_classes, reduction="sum")
    print(loss)
