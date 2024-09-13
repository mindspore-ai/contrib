# import torch
from mindspore import Tensor
import numpy as np
import cv2
import os
from util.config import config as cfg
import mindspore.ops as ops


def visualize_gt(image, contours, tr=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)
    # tr = Tensor(tr)

    if tr is not None:
        for i in range(3):
            h, w = tr[i].shape
            scale = 8 * (2 ** i)
            tr[i] = Tensor(tr[i])
            tr[i] = tr[i].reshape(1, 1, h, w)
            interpolate = ops.ResizeNearestNeighbor((scale * h, scale * w))
            tr[i] = interpolate(tr[i]).reshape(h * scale,w * scale)
            tr[i] = tr[i].asnumpy()
            # tr[i] = torch.nn.functional.interpolate(tr[i].view(1, 1, h, w), scale_factor=scale).view(h * scale,
            #                                                                                          w * scale).cpu().numpy()


        tr = ((tr[0] > cfg.tr_thresh) | (tr[1] > cfg.tr_thresh) | (tr[2] > cfg.tr_thresh)).astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        image_show = np.concatenate([image_show, tr], axis=1)
        return image_show
    else:
        return image_show


def visualize_detection(image, contours, tr=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    cv2.drawContours(image_show, contours, -1, (0, 0, 255), 2)

    if tr is not None:
        for i in range(3):
            h, w = tr[i].shape
            scale = 8 * (2 ** i)
            tr[i] = Tensor(tr[i])
            tr[i] = tr[i].reshape(1, 1, h, w)
            interpolate = ops.ResizeNearestNeighbor((scale * h, scale * w))
            tr[i] = interpolate(tr[i]).reshape(h * scale,w * scale)
            tr[i] = tr[i].asnumpy()
            # tr[i] = torch.tensor(tr[i])
            # tr[i] = torch.nn.functional.interpolate(tr[i].view(1,1,h,w), scale_factor = scale).view(h*scale,w*scale).cpu().numpy()

        tr = ((tr[0] > cfg.tr_thresh) | (tr[1] > cfg.tr_thresh) | (tr[2] > cfg.tr_thresh)).astype(np.uint8)
        tr = cv2.cvtColor(tr * 255, cv2.COLOR_GRAY2BGR)
        image_show = np.concatenate([image_show, tr], axis=1)
        return image_show
    else:
        return image_show