import mindspore
from mindspore import ops, nn, Tensor
from mindspore.experimental.optim.optimizer import Optimizer

class AccSGD(Optimizer):

    def __init__(self, params, lr, kappa = 1000.0, xi = 10.0, smallConst = 0.7, weight_decay = 0):
        defaults = dict(lr = lr, kappa = kappa, xi = xi, smallConst = smallConst,
                        weight_decay = weight_decay)
        super(AccSGD, self).__init__(params, defaults)
        
    def construct(self, gradients):
        for group_id, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            large_lr = (group['lr'] * group['kappa']) / (group['smallConst'])
            Alpha = 1.0 - ((group['smallConst'] * group['smallConst'] * group['xi']) / group['kappa'])
            Beta = 1.0 - Alpha
            zeta = group['smallConst'] / (group['smallConst'] + Beta)
            id = self.group_start_id[group_id]
            for i, p in enumerate(group['params']):
                if gradients[id+i] is None:
                    continue
                d_p = gradients[id+i]
                if weight_decay != 0:
                    d_p = ops.add(ops.add(d_p, weight_decay), p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = p.copy()
                buf = param_state['momentum_buffer']
                buf = buf.mul((1.0 / Beta) - 1.0)
                buf = buf.add(-large_lr)
                buf = buf.add(d_p)
                buf = buf.add(p.data)
                buf = buf.mul(Beta)

                p.set_data(ops.add(ops.add(p.data, -group['lr']), d_p))
                p.set_data(p.data.mul(zeta))
                p.set_data(ops.add(ops.add(p.data, 1.0 - zeta), buf))
        return True

if __name__ == '__main__':
    import numpy as np
    
    net = nn.Dense(8, 2)
    data = Tensor(np.random.rand(20, 8).astype(np.float32))
    label = Tensor(np.random.rand(20, 2).astype(np.float32))
    optimizer = AccSGD(net.trainable_params(), 1e-4)
    criterion = nn.MAELoss(reduction = 'mean')

    def forward_fn(data, label):
        logits = net(data)
        loss = criterion(logits, label)
        return loss, logits
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux = True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        print(loss)

    for i in range(10):
        train_step(data, label)