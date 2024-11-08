import mindspore as ms
from mindspore import Parameter, ops, nn
import numpy as np

from loss import loss_EPM

class GraphConv(nn.Cell):
    def __init__(self,
                 hid_dim=2048):
        super(GraphConv, self).__init__()
        self.hid_dim = hid_dim
        # ?
        self.W = Parameter(ops.zeros((hid_dim, hid_dim), dtype=ms.float32), requires_grad=True)
        self.act = nn.ReLU()
        
    def construct(self, x, A):
        b, t, c = x.shape
        W = self.W.unsqueeze(0).tile((b, 1, 1))
        o = ops.bmm(A, x)
        o = ops.bmm(o, W)
        return o
    
class ACGNet(nn.Cell):
    def __init__(self,
                 num_layers=2,
                 hid_dim=2048):
        super(ACGNet, self).__init__()
        self.hid_dim = hid_dim
        self.gcn = nn.CellList(
            [GraphConv(hid_dim) for _ in range(num_layers)]
        )
    
    def make_ssg(self, x):
        '''
            make similarity graph
            input: feature(x) : b, t, c
            output: ssg: b, t, t
        '''
        x_t = x.permute(0, 2, 1)
        
        x_norm = ops.norm(x, dim=2, keepdim=True) + 1e-9
        x_t_norm = ops.norm(x_t, dim=1, keepdim=True) + 1e-9
        
        ssg = ops.bmm((x/x_norm), (x_t/x_t_norm))
        return ssg
    
    def make_tdg(self, x, Z=10):
        '''
            make temporal diffusion graph
            input: x: b, t, c
                   Z: hyperparameter, an integer indicating diffusion degree 
        '''
        b, t, c = x.shape
        tmp_vecs = [ops.arange(t, dtype=ms.float32) - i for i in range(t)]
        g = ops.abs(ops.stack(tmp_vecs))
        g = ops.maximum(Z-g, ops.zeros_like(g))
        tgd = 1. - g / Z
        tgd = tgd.unsqueeze(0).tile((b, 1, 1))
        return tgd
    
    def make_osg(self, x, alpha=1., lamda=0.85, K=50):
        '''
            make overall sparse graph
        '''
        b, t, c = x.shape
        A = 0.5 * (self.make_ssg(x) + alpha * self.make_tdg(x))
        _, idx = ops.topk(A, k=K, dim=-1, largest=True)
        mask = ops.zeros_like(A)
        # ?
        mask = mask.scatter(axis=-1, index=idx, src=1.)
        # mask = ms.ops.scatter(mask, axis=-1, index=idx, src=ms.Tensor(np.ones_like(mask)))

        A = A * mask
        A = ops.where(A > lamda, A, ops.full_like(A, fill_value=0, dtype=A.dtype))
        return A
    
    def construct(self, x):
        A_prime = self.make_osg(x)
        A_hat = A_prime / (ops.sum(A_prime, dim=-1, keepdim=True) + 1e-9)
        F_avg = ops.bmm(A_hat, x)
        for i, layer in enumerate(self.gcn):
            if i == 0:
                F_conv = layer(x, A_hat)
            else:
                F_conv = layer(F_conv, A_hat)
        return x, x + F_avg + F_conv, A_prime