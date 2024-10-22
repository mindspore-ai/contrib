import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Tensor
import numpy as np


def cosine_similarity_maps(model: nn.Cell, X: Tensor, sign: bool = True, rescale: bool = True) -> Tensor:
    deltas = []
    X_grad = X.copy()

    logits = model(X_grad)

    if rescale:
        logits = logits / ops.ReduceMax(True)(ops.Abs()(logits), 1) * 10

    B = logits.shape[0]
    classes = logits.shape[-1]

    for c in range(classes):
        y = Tensor(np.ones((B,)), mindspore.int32) * c
        loss = nn.SoftmaxCrossEntropyWithLogits()(logits, ops.OneHot()(y, classes, Tensor(1.0, mindspore.float32),
                                                                       Tensor(0.0, mindspore.float32)))
        loss = loss.mean()
        grad_fn = ops.GradOperation(get_by_list=True, sens_param=False)
        grad = grad_fn(model, X_grad)(X_grad)

        if sign:
            grad = ops.Sign()(grad)

        deltas.append(grad.copy())
        X_grad = ops.ZerosLike()(X_grad)

    model.set_train(False)
    deltas = mnp.stack(deltas, axis=0)

    deltas = ops.ReduceMax()(deltas, axis=-3)

    deltas = deltas.view(classes, B, -1)
    norm = ops.norm(deltas, ord=2, dim=2, keepdim=True)
    deltas = deltas / norm

    deltas = deltas.transpose(0, 2, 1)

    csm = ops.matmul(deltas, deltas.transpose(0, 2, 1))

    if ops.IsNan()(csm).any():
        raise Exception("NaNs in CSM!")

    return csm