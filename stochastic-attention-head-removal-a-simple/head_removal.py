import random
import mindspore
from mindspore import ops

def remove_head(p, mha, istrain):
    if not istrain:
        return mha

    batch, head, time, d_k = mha.shape

    # dropout probability is p
    if random.random() < p:
        mask = ops.ones([batch, 1, time, d_k], dtype=mindspore.uint8)  # masked_fill fill ones
    else:
        mask = ops.zeros([batch, 1, time, d_k], dtype=mindspore.uint8)

    for _ in range(head - 1):
        if random.random() < p:
            maskH = ops.ones([batch, 1, time, d_k], dtype=mindspore.uint8)  # masked_fill fill ones
        else:
            maskH = ops.zeros([batch, 1, time, d_k], dtype=mindspore.uint8)
        mask = ops.cat((mask, maskH), dim=1)

    mha = mha.masked_fill(mask, 0)
    return mha * (1 / (1 - p))