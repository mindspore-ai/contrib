import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import cv2
import numpy as np



class Integrated_Gradient:
    def __init__(self, model, preprocess, steps=20):     
        self.model = model
        self.prep = preprocess
        self.steps = steps

    def get_heatmap(self, img, baseline=None):        
        # check whether the input shape is the same as the baseline shape
        x = Tensor(self.prep(img),mindspore.float32)
        if baseline is None:
            baseline = ops.zeros_like(x, dtype=mindspore.float32)
        self._check(x, baseline)
        
        # get predict label
        output = self.model(x)
        pred_label = output.max(axis = 1,return_indices=True)[1]
        
        # compute integrated gradients
        X, delta_X = self._get_X_and_delta(x, baseline, self.steps)

        def get_grad(X_0,label):
            return self.model(X_0)[:,label].sum()
        
        grads = mindspore.grad(get_grad,grad_position=0)(X,pred_label)
        grads = grads.asnumpy()
        grads = delta_X.asnumpy() * (grads[:-1] + grads[1:]) / 2.
        ig_grad = grads.sum(axis=0, keepdims=True)
        
        # plot
        cam = np.clip(ig_grad, 0, None)
        cam = cam / cam.max() * 255.
        cam = cam.squeeze().transpose(1, 2, 0).astype(np.uint8)
        cam = cv2.resize(cam, img.size[:2])
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_TURBO)[..., ::-1]      
        
        return self.model(ops.stop_gradient(x)), cam
                                         
    def _get_X_and_delta(self, x, baseline, steps):
        alphas = ops.linspace(Tensor(0,mindspore.float32), Tensor(1,mindspore.float32), steps + 1).view(-1, 1, 1, 1)
        delta = (x - baseline)
        x = baseline + alphas * delta
        return x, ops.stop_gradient(delta / steps)
        
    def _check(self, x, baseline):
        if x.shape != baseline.shape:
            raise ValueError(f'input shape should equal to baseline shape. '
                             f'Got input shape: {x.shape}, '
                             f'baseline shape: {baseline.shape}') 

