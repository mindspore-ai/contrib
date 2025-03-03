import mindspore.ops as ops


def enhanced_mixing_loss(y_true, y_pred):
    # Code written by Seung hyun Hwang
    gamma = 1.1
    alpha = 0.48
    smooth = 1.0
    epsilon = 1e-7
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    # dice loss
    intersection = (y_true * y_pred).sum()
    dice_loss = (2.0 * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

    # focal loss
    y_pred = ops.clamp(y_pred, epsilon)

    pt_1 = ops.where(ops.equal(y_true, 1), y_pred, ops.ones_like(y_pred))
    pt_0 = ops.where(ops.equal(y_true, 0), y_pred, ops.zeros_like(y_pred))
    focal_loss = -ops.mean(alpha * ops.pow(1.0 - pt_1, gamma) * ops.log(pt_1)) - \
                 ops.mean((1 - alpha) * ops.pow(pt_0, gamma) * ops.log(1.0 - pt_0))
    return focal_loss - ops.log(dice_loss)
