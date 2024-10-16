import mindspore
from mindspore import nn,ops


class Dice_Loss(nn.Cell):
    """
    Calculates the Sørensen–Dice coefficient-based loss.
    Taken from
    https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/loss.py#L28

    Args:
        inputs (mindspore.Tensor): 1-hot encoded predictions
        targets (mindspore.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super().__init__()

    def construct(self, inputs, targets):
        """
        Dice(A, B) = (2 * |intersection(A, B)|) / (|A| + |B|)
        where |x| denotes the cardinality of the set x.
        """
        mul = ops.mul(inputs, targets)
        add = ops.add(inputs, targets)
        dice = 2 * ops.div(mul.sum(), add.sum())
        return 1 - dice


class MCC_Loss(nn.Cell):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.

    Args:
        inputs (mindspore.Tensor): 1-hot encoded predictions
        targets (mindspore.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super().__init__()

    def construct(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = ops.sum(ops.mul(inputs, targets))
        tn = ops.sum(ops.mul((1 - inputs), (1 - targets)))
        fp = ops.sum(ops.mul(inputs, (1 - targets)))
        fn = ops.sum(ops.mul((1 - inputs), targets))

        numerator = ops.mul(tp, tn) - ops.mul(fp, fn)
        denominator = ops.sqrt(
            ops.add(tp, fp)
            * ops.add(tp, fn)
            * ops.add(tn, fp)
            * ops.add(tn, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = ops.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc