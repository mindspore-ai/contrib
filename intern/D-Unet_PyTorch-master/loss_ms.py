import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn

def enhanced_mixing_loss(y_true, y_pred):
    gamma = 1.1
    alpha = 0.48
    smooth = 1.
    epsilon = 1e-7
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # Dice loss
    intersection = (y_true_flat * y_pred_flat).sum()
    dice_score = (2. * intersection + smooth) / (
        (y_true_flat * y_true_flat).sum() + (y_pred_flat * y_pred_flat).sum() + smooth
    )
    dice_loss = 1. - dice_score
    
    # Focal loss
    y_pred_clipped = ops.clip_by_value(y_pred_flat, epsilon, 1. - epsilon)
    
    pt_1 = ops.select(ops.equal(y_true_flat, 1.), y_pred_clipped, ops.ones_like(y_pred_clipped))
    pt_0 = ops.select(ops.equal(y_true_flat, 0.), y_pred_clipped, ops.zeros_like(y_pred_clipped))
    
    focal_loss_1 = alpha * ops.pow(1. - pt_1, gamma) * ops.log(pt_1)
    focal_loss_0 = (1. - alpha) * ops.pow(pt_0, gamma) * ops.log(1. - pt_0)
    
    focal_loss = -(focal_loss_1.mean() + focal_loss_0.mean())
    total_loss = focal_loss + dice_loss
    
    return total_loss