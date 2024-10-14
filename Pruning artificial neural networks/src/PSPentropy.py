import numpy as np
import mindspore as ms
from mindspore import nn, ops

class Hook():
    def __init__(self, module, backward=False):
        self.module = module[1]
        self.name = module[0]
        self.input = None
        self.output = None
        
    def hook_fn(self, cell, inputs, outputs):
        self.input = inputs
        self.output = outputs

    def register(self):
        self.hook = self.module.register_forward_hook(self.hook_fn)

    def close(self):
        self.hook.remove()

def compute_entropy_1st(model, dataset, hookF, classes):
    H = [None] * len(hookF)
    H_classwise = [None] * len(hookF)
    P = [None] * len(hookF)
    N = [None] * len(hookF)
    M = ms.Tensor(np.zeros(classes), ms.float32)

    for idx in range(len(hookF)):
        output_shape = hookF[idx].output.shape
        H_classwise[idx] = ms.Tensor(np.zeros((np.prod(output_shape[1:]), classes)), ms.float32)
        P[idx] = ms.Tensor(np.zeros((np.prod(output_shape[1:]), classes)), ms.float32)
        N[idx] = ms.Tensor(np.zeros((np.prod(output_shape[1:]), classes)), ms.float32)

    for data in dataset.create_dict_iterator():
        xb = data['image']
        yb = data['label']
        for this_idx_1 in range(classes):
            M[this_idx_1] += ops.sum(yb == this_idx_1).asnumpy()
        
        model(xb)
        
        for idx in range(len(hookF)):
            for this_idx_1 in range(classes):
                this_yb_1 = ops.cast(yb == this_idx_1, ms.float32).expand_dims(1)
                output_reshaped = hookF[idx].output.view((this_yb_1.shape[0], -1))
                P[idx][:, this_idx_1] += ops.sum((output_reshaped > 0) * this_yb_1, axis=0)
                N[idx][:, this_idx_1] += ops.sum((output_reshaped <= 0) * this_yb_1, axis=0)

    for idx in range(len(hookF)):
        P[idx] = ops.clip_by_value(P[idx] / M.expand_dims(0), 0.0001, 0.9999)
        N[idx] = ops.clip_by_value(N[idx] / M.expand_dims(0), 0.0001, 0.9999)
        for this_idx_1 in range(classes):
            H_classwise[idx][:, this_idx_1] -= (P[idx][:, this_idx_1] * ops.log2(P[idx][:, this_idx_1]) + 
                                                N[idx][:, this_idx_1] * ops.log2(N[idx][:, this_idx_1]))
        H[idx] = ops.sum(H_classwise[idx], axis=1)
    
    return H, H_classwise

def compute_PSPentropy(model, dataset, order=1, classes=10):
    hookF = [Hook(layer, backward=False) for layer in model.name_cells().items()]
    for h in hookF:
        h.register()

    for data in dataset.create_dict_iterator():
        inputs = data['image']
        model(inputs)
        break

    if order == 1:
        H, H_classwise = compute_entropy_1st(model, dataset, hookF, classes)
        for h in hookF:
            h.close()
        return H, H_classwise
    else:
        raise NotImplementedError("Orders other than 1 are not yet implemented")
