import mindspore
from mindspore import Tensor, ops
import numpy as np

def loss_function(x, problem='zdt1'):
    
    ### x = (100, 30)
    f = x[:, 0]
    g = x[:, 1:]

    if problem == 'zdt1':
        g = g.sum(axis=1, keepdims=False) * (9./29.) + 1.
        h = 1. - ops.sqrt(f/g)
    
    if problem == 'zdt2':
        g = g.sum(axis=1, keepdims=False) * (9./29.) + 1.
        h = 1. - (f/g)**2

    if problem == 'zdt3':
        g = g.sum(axis=1, keepdims=False) * (9./29.) + 1.
        h = 1. - ops.sqrt(f/g) - (f/g)*ops.sin(10.*np.pi*f)


    return f, g*h

def get_ref_point(problem='zdt1'):
    if problem == 'zdt1':
        return np.array([0.99022638, 6.39358545])

    if problem == 'zdt2':
        return np.array([0.99022638, 7.71577261])

    if problem == 'zdt3':
        return np.array([0.99022638, 6.54635266])

