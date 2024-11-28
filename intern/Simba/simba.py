import mindspore as ms
from mindspore.nn import Optimizer
from mindspore import ops,Tensor
import math

class Simba(Optimizer):
    """
    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.01)
        momentum (float, optional) -- coefficient used for computing running averages of gradient (default: 0.9)
        coarse_dim_perc (float, optional) -- number of coarse model dimensions in percentage of the fine model dimensions (default: 0.5)
        rank (int, optional) -- number of eigevalues and eigenvectors to be computed by the T-SVD
        eps (float, optional) -- lower bound on eigenvalues to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
    """ 

    def __init__(self,params,learning_rate=1e-2,momentum=0.9,coarse_dim_perc=0.5,rank=10,eps=1e-8,weight_decay=0.):
            super().__init__(learning_rate, params)
            self.learning_rate=learning_rate
            self.momentum=momentum
            self.coarse_dim_perc=coarse_dim_perc
            self.rank=rank
            self.eps=eps
            self.weight_decay=weight_decay
            self.grad_avgs=self.parameters.clone(prefix="grad_avgs", init='zeros')
    

    def construct(self,gradients):
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)

        grads=gradients
        grad_avgs=self.grad_avgs


        updated_params=self.simba(params=self.parameters, grads=grads, grad_avgs=grad_avgs, 
        momentum=self.momentum, coarse_dim_perc=self.coarse_dim_perc, rank=self.rank, weight_decay=self.weight_decay, eps=self.eps, lr=self.learning_rate)

        
        return updated_params


    def simba(self,params,grads,grad_avgs,momentum,coarse_dim_perc,rank,weight_decay,eps,lr):
        
        updated_params=[]

        for i,param in enumerate(params):
            grad=grads[i]
            grad_avg=grad_avgs[i]

            if weight_decay!=0:
                grad=grad.add(param.mul(weight_decay))
            
            grad_avg=grad_avg.mul(momentum).add(grad)

            #mindspore2.3要求randperm的参数是64位整数
            n_weights=Tensor(grad.shape[0],ms.int64)

            #随机选取corse_dim计算粗略移动平均梯度grad_avg_sub(G_l,k)
            coarse_dim=math.ceil(coarse_dim_perc*n_weights)+1
            coarse_dim=ms.Tensor(coarse_dim,dtype=ms.int32)
            idx=ops.randperm(n_weights)[:coarse_dim]
            grad_avg_mat=grad_avg.view(param.shape[0],-1)
            grad_avg_sub=grad_avg_mat[idx]

            #对预调节器做SVD分解，为降低运算复杂度，对维度高于rank的采用低秩分解
            precond_reduced=grad_avg_sub @ grad_avg_sub.t()
            if rank<precond_reduced.shape[0]:
                S,U,V=ops.svd(precond_reduced)
                U_r = U[:, :rank]
                Sigma_r = S[:rank]
                v = V[:rank, :]
            else:
                Sigma_r,U_r,v=ops.svd(precond_reduced)            
            
            Sigma_r=Sigma_r.abs().sqrt()
            if eps!=0:
                Sigma_r[Sigma_r<eps]=eps
            
            #去掉最后一列
            U_r_minus1 = U_r[:, :-1]
            Sigma_minus1_inv = ops.diag(Sigma_r[:-1] ** (-1) - Sigma_r[-1] ** (-1))

            #计算Q_l,k^(-1/2)G_l,k
            a_term=U_r_minus1.t()@grad_avg_sub
            b_term=Sigma_minus1_inv@a_term
            c_term=U_r_minus1@b_term
            dH=-grad_avg_sub/Sigma_r[-1]-c_term

            d=ops.zeros_like(grad_avg_mat)
            d[idx]=dH
            d_unflat=d.view(param.shape)

            #更新参数
            updated_param=param.add(d_unflat.mul(lr))
            ops.assign(self.parameters[i],updated_param)
            updated_params.append(param)

        return updated_params
        
            
     


