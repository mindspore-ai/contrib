'''
All-In-One Adam Optimizer where several novelties are combined from following papers:

    Decoupled Weight Decay Regularization for Adam https://arxiv.org/abs/1711.05101

Authors shown that the real reason why Momentum optimizer is often outperforming Adam in generalization was due to 
the fact that Adam does not perform well under L2 regularization and developed decoupled weight decay as a solution.

    Online Learning Rate Adaptation with Hypergradient Descent https://arxiv.org/abs/1703.04782

This is enabled via "hypergrad" parameter by setting it to any value except zero. It enables the optimizer to update
the learning-rate itself by the technique proposed in the paper, instead of giving an external schedule which would 
require lots of additional hyperparameters. It is especially useful when one doesn't want to hypertune a schedule.

    Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks 
    https://arxiv.org/abs/1711.05101

This can be set by the "partial" parameter, which controls how likely the optimizer acts similar to Adam (1.0) and 
SGD (0.0), which is very useful if hypertuned. One can also update (decay) this parameter online to switch between 
Adam and SGD optimizers in an easy way, which has been recommended by previous research for a better generalization.
'''

import math
from mindspore import nn, ops
from mindspore.experimental import optim

class AdamAIO(optim.Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=5e-3, weight_decay=5e-6, hypergrad=1e-7, partial=0.75):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hypergrad=hypergrad, partial=partial)
        super().__init__(params, defaults)
    
    def construct(self, gradients, closure=None):
        loss = None if closure is None else closure()
        for group_id, group in enumerate(self.param_groups):
            id = self.group_start_id[group_id]
            for i, p in enumerate(group['params']):
                if gradients[id+i] is None: continue
                grad = gradients[id+i]
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = ops.zeros_like(p.data)
                    state['exp_avg_sq'] = ops.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['hypergrad'] > 0 and state['step'] > 1:
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    h = ops.tensor_dot(grad.view(-1), ops.div(exp_avg, exp_avg_sq.sqrt().add(group['eps'])).view(-1), axes=1) * math.sqrt(prev_bias_correction2) / prev_bias_correction1
                    group['lr'] += group['hypergrad'] * h
                exp_avg = exp_avg.mul(beta1).add((1 - beta1) * grad)
                exp_avg_sq = exp_avg_sq.mul(beta2).add((1 - beta2) * grad * grad)
                denom = exp_avg_sq.sqrt().add(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    decayed_weights = ops.mul(p.data, group['weight_decay'])
                    p.set_data(p.data.addcdiv(-step_size, exp_avg, denom**group['partial']))
                    p.set_data(p.data.sub(decayed_weights))
                else:
                    p.set_data(p.data.addcdiv(-step_size, exp_avg, denom**group['partial']))
        return loss
    

if __name__ == '__main__':
    import numpy as np
    import mindspore as ms
    from mindspore import Tensor
    
    net = nn.Dense(8, 2)
    data = Tensor(np.random.rand(20, 8).astype(np.float32))
    label = Tensor(np.random.rand(20, 2).astype(np.float32))
    optimizer = AdamAIO(net.trainable_params(), 0.01)
    criterion = nn.MAELoss(reduction="mean")

    def forward_fn(data, label):
        logits = net(data)
        loss = criterion(logits, label)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        print(loss)

    for i in range(10):
        train_step(data, label)
