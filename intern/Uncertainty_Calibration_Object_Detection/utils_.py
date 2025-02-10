# -*- coding: utf-8 -*-
"""
Migrated to MindSpore on [current date]

@author: [your name]
"""
import mindspore
import numpy as np
import os
import time
from mindspore import Tensor, ops

def text_to_mindspore(txt_file):
    array = np.loadtxt(txt_file)
    if array.ndim == 2:
        return Tensor(array)
    if array.ndim == 1:
        return ops.expand_dims(Tensor(array), 0)

def IoU(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = ops.maximum(box1_x1, box2_x1)
    y1 = ops.maximum(box1_y1, box2_y1)
    x2 = ops.minimum(box1_x2, box2_x2)
    y2 = ops.minimum(box1_y2, box2_y2)

    # Clip to ensure non-negative values
    intersection = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    
    box1_area = ops.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = ops.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def fill_blank_preds(labels_folder, preds_folder):
    l1 = os.listdir(labels_folder)
    l2 = os.listdir(preds_folder)
    res = [x for x in l1 if x not in l2]
    for file in res:
        with open(os.path.join(preds_folder, file), 'w') as f:
            f.write('-1 0.0 0.0 0.0 0.0 0.0')

def divide_detections(labels_folder, preds_folder, iou_thresh=0.5):
    fill_blank_preds(labels_folder, preds_folder)
    TP = []
    FP = []
    FN_count = 0
    
    for filename in os.listdir(labels_folder):
        TP_file = []
        
        labels_file = os.path.join(labels_folder, filename)
        preds_file = os.path.join(preds_folder, filename)
        
        labels_tensor = text_to_mindspore(labels_file)
        preds_tensor = text_to_mindspore(preds_file)
        
        # TRUE POSITIVES and FALSE NEGATIVES
        for i in range(labels_tensor.shape[0]):
            idx = -1
            max_iou = 0
            for j in range(preds_tensor.shape[0]):
                if labels_tensor[i, 0] == preds_tensor[j, 0]:
                    current_iou = IoU(labels_tensor[i, 1:5], preds_tensor[j, 1:5])
                    if current_iou > max_iou and current_iou >= iou_thresh:
                        max_iou = current_iou
                        idx = j
            if idx < 0:
                FN_count += 1
            else:
                TP_file.append(np.array([preds_tensor[idx, 5].asnumpy(), max_iou.asnumpy(), idx]))
        
        TP.extend(TP_file)
        
        # FALSE POSITIVES
        for i in range(preds_tensor.shape[0]):
            matched = False
            for entry in TP_file:
                if i == entry[2]:
                    matched = True
                    break
            if not matched and preds_tensor[i, 0] != -1:
                FP.append(np.array([preds_tensor[i, 5].asnumpy()]))
    
    return TP, FP, FN_count
