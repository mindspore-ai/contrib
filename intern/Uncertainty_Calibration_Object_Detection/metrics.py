# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:12:04 2023

@author: pedro
"""

import numpy as np
import os
from utils_ import divide_detections

def QGC(labels_folder, preds_folder, iou_thresh=0.5):
    
    TP, FP, FN_count = divide_detections(labels_folder, preds_folder, iou_thresh=iou_thresh)
    
    TP = TP[:,0].astype(float)
    FP = FP.astype(float)
    FN_count = FN_count[0]
    
    tp_score = 0.0
    fp_score = 0.0
    for i in range(len(TP)):
        tp_score += (1-TP[i])**2
    for i in range(len(FP)):
        fp_score += FP[i]**2
    
    QGC_total = tp_score.item() + fp_score.item() + FN_count
    QGC_avg = (tp_score.item() + fp_score.item() + FN_count) / (len(TP) + len(FP) + FN_count)
    
    return QGC_total, QGC_avg



def SGC(labels_folder, preds_folder, iou_thresh=0.5):
    
    TP, FP, FN_count = divide_detections(labels_folder, preds_folder, iou_thresh=iou_thresh)
    
    TP = TP[:,0].astype(float)
    FP = FP.astype(float)
    FN_count = FN_count[0]
    
    tp_score = 0
    fp_score = 0
    for i in range(len(TP)):
        tp_score += TP[i] / (TP[i]**2 + (1-TP[i])**2)**(1/2)
    for i in range(len(FP)):
        fp_score += (1-FP[i]) / (FP[i]**2 + (1-FP[i])**2)**(1/2)
      
    SGC_total = len(TP) + len(FP) + FN_count - tp_score.item() - fp_score.item()
    SGC_avg = (len(TP) + len(FP) + FN_count - tp_score.item() - fp_score.item()) / (len(TP) + len(FP) + FN_count)
    
    return SGC_total, SGC_avg




def EGCE(labels_folder, preds_folder, num_bins=15, iou_thresh=0.5, conf_thresh=0.1):
         
    TP, FP, FN_count = divide_detections(labels_folder, preds_folder, iou_thresh=iou_thresh)
    
    TP = TP[:,0].astype(float)
    FP = FP.astype(float)
    FN_count = FN_count[0]
    
    TP_new = TP[TP>=conf_thresh]
    FP = FP[FP>=conf_thresh]
    
    FN_count = FN_count + (len(TP)-len(TP_new))
    
    TP=TP_new
    
    bins = np.linspace(0.0, 1.0, num_bins+1)
    bins[0] = bins[0]-0.001

    TP_binned = np.digitize(TP, bins, right=True)-1
    FP_binned = np.digitize(FP, bins, right=True)-1
    
    EGCE = 0
    
    bin_precs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    
    for i in range(num_bins):
        if i == num_bins-1:
            bin_sizes[i] = len(TP_binned[TP_binned == i]) + len(FP_binned[FP_binned == i]) + FN_count
        else:
            bin_sizes[i] = len(TP_binned[TP_binned == i]) + len(FP_binned[FP_binned == i])
        
        if bin_sizes[i] > 0:
          
          if i == num_bins-1:
              bin_precs[i] = len(TP_binned[TP_binned == i]) / bin_sizes[i]
              bin_confs[i] = ( (TP[TP_binned==i]).sum() + (FP[FP_binned==i]).sum() + FN_count)  / bin_sizes[i] 
          else:
              bin_precs[i] = len(TP_binned[TP_binned == i]) / bin_sizes[i]
              bin_confs[i] = ( (TP[TP_binned==i]).sum() + (FP[FP_binned==i]).sum() )  / bin_sizes[i]
          
    for i in range(num_bins):
      abs_conf_dif = abs(bin_precs[i] - bin_confs[i])
      EGCE += (bin_sizes[i] ) * abs_conf_dif
      
    
    EGCE_avg = EGCE/sum(bin_sizes)
      
    return EGCE, EGCE_avg
    

    
def DECE(labels_folder, preds_folder, num_bins=15, iou_thresh=0.5, conf_thresh=0.1):
         
    TP, FP, _ = divide_detections(labels_folder, preds_folder, iou_thresh=iou_thresh)
    
    TP = TP[:,0].astype(float)
    FP = FP.astype(float)
    
    TP = TP[TP>=conf_thresh]
    FP = FP[FP>=conf_thresh]
    
    bins = np.linspace(0.0, 1.0, num_bins+1)
    bins[0] = bins[0]-0.001

    TP_binned = np.digitize(TP, bins, right=True)-1
    FP_binned = np.digitize(FP, bins, right=True)-1
    
    DECE = 0
    
    bin_precs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_sizes[i] = len(TP_binned[TP_binned == i]) + len(FP_binned[FP_binned == i])
        if bin_sizes[i] > 0:
            bin_precs[i] = len(TP_binned[TP_binned == i]) / bin_sizes[i]
            bin_confs[i] = ( (TP[TP_binned==i]).sum() + (FP[FP_binned==i]).sum() )  / bin_sizes[i]
          
    for i in range(num_bins):
      abs_conf_dif = abs(bin_precs[i] - bin_confs[i])
      DECE += (bin_sizes[i] ) * abs_conf_dif
      
    
    DECE_avg = DECE/sum(bin_sizes)
      
    return DECE, DECE_avg



if __name__ == "__main__":

    labels = # labels directory
    detections =# detections directory
    
    qgc, qgc_avg = QGC(labels, detections, iou_thresh=0.5)
    
    print(qgc,qgc_avg)
    
    
    
    
