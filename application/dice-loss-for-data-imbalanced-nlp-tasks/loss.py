import mindspore as ms
from mindspore import nn, ops


class SelfAdjDiceLoss(nn.Cell):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def construct(self, logits: ms.Tensor, targets: ms.Tensor) -> ms.Tensor:
        probs = ops.softmax(logits, axis=1)
        probs = ops.gather_d(probs, dim=1, index=targets.expand_dims(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")

if __name__ == "__main__":
    criterion = SelfAdjDiceLoss(reduction="none")
    # (batch_size, num_tokens, num_classes)
    logits = ops.rand(128, 40, 10)
    targets = ops.randint(0, 10, size=(128, 40))

    loss = criterion(logits.reshape(-1, 10), targets.reshape(-1))
    loss = loss.reshape(-1, 40).mean(-1).mean()
    print(loss)