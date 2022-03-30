#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao CHEN (chaochancs@gmail.com)
# Created On: 2017-08-11
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Parameter, Tensor
import numpy as np

class FocalLoss(nn.Cell):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Parameter(mindspore.numpy.ones(class_num, 1))
        else:
            if isinstance(alpha, Parameter):
                self.alpha = alpha
            else:
                self.alpha = Parameter(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        print(N)
        C = inputs.size(1)
        P = nn.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Parameter(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        

        '''
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        '''
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(ops.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        
        if self.size_average:
            loss = mindspore.numpy.mean(batch_loss)
        else:
            loss = mindspore.numpy.sum()
        return loss

        

if __name__ == "__main__":
    alpha = Tensor(np.random.rand(21, 1))
    print(alpha)
    FL = FocalLoss(class_num=5, gamma=0 )
    CE = nn.transformer.CrossEntropyLoss()
    N = 4
    C = 5
    inputs = Tensor(np.random.rand(N, C))
    targets = Tensor(np.random.randint(C, N))
    inputs_fl = Parameter(inputs.clone(), requires_grad=True)
    targets_fl = Parameter(targets.clone())

    inputs_ce = Parameter(inputs.clone(), requires_grad=True)
    targets_ce = Parameter(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)
    
        


