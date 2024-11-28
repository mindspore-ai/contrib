import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

    
def normalize(d):
    d /= (ops.sqrt(ops.sum(d**2, axis=1)).view(-1,1)+1e-16)
    return d

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= ops.norm(d_reshaped, dim=1, keepdim=True) + 1e-16
    return d

def _kl_div(p,q):
    '''
    D_KL(p||q) = Sum(p log p - p log q)
    '''
    logp = ops.log_softmax(p, axis=1)
    logq = ops.log_softmax(q, axis=1)
    p = ops.exp(logp)
    return (p*(logp-logq)).sum(axis=1, keepdims=True).mean()

class VATLoss(nn.Cell):
    def __init__(self, xi =0.001, eps=0.1, ip=2):
        """
        VAT loss
        :param xi: hyperparameter of VAT.  a small float for the approx. of the finite difference method.
        :param eps: hyperparameter of VAT. the value for how much deviate from original data point X.
        :param ip: a number of power iteration for approximation of r_vadv. The default value 2 is sufficient.
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        
    def construct(self, model, x):
        pred = model(x)
        
        # random unit for perturbation
        d = ops.randn(x.shape)
        d = _l2_normalize(d)
        
        def adv_distance_fn(d):
            pred_hat = model(x + self.xi * d)
            return _kl_div(pred_hat, pred)
        
        for _ in range(self.ip):
            grad_fn = ms.grad(adv_distance_fn)
            grad_d = grad_fn(d)
            
            d = _l2_normalize(grad_d)
        
        r_adv = d * self.eps
        pred_hat = model(x + r_adv)
        lds = _kl_div(pred_hat, pred)
        return lds


if __name__ == '__main__':
    model = nn.Dense(10, 5)
    x = ops.randn(3, 10)
    vat_loss = VATLoss()
    print(vat_loss(model, x))
