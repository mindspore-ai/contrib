# coding=utf-8
# adapted from:
# https://github.com/mindspore-ai/mindspore/blob/master/model_zoo/official/cv/deeplabv3/src/utils/learning_rates.py

import numpy as np

__all__ = ['lr_scheduler']


def lr_scheduler(lr_type, base_lr, total_train_steps,
                 lr_decay_step=None,
                 lr_decay_rate=None):
    assert lr_type in ('cos', 'poly', 'exp')
    if lr_type == 'cos':
        lr_iter = _cosine_lr(base_lr, total_train_steps, total_train_steps)
    elif lr_type == 'poly':
        lr_iter = _poly_lr(base_lr, total_train_steps, total_train_steps, end_lr=.0, power=.9)
    else:
        lr_iter = _exponential_lr(base_lr, lr_decay_step, lr_decay_rate, total_train_steps,
                                  staircase=True)
    return lr_iter


def _cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def _poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr


def _exponential_lr(base_lr, decay_steps, decay_rate, total_steps, staircase=False):
    for i in range(total_steps):
        if staircase:
            power_ = i // decay_steps
        else:
            power_ = float(i) / decay_steps
        yield base_lr * (decay_rate ** power_)
