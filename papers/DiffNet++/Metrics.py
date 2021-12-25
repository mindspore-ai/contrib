"""Metrics"""
#import numpy as np
import math

def Getidcg(length):
    """Get idcg value"""
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg

def Getdcg(idx):
    """Get dcg value"""
    dcg = math.log(2) / math.log(idx + 2)
    return dcg

def Gethr():
    """Get hit radio"""
    hit = 1.0
    return hit
