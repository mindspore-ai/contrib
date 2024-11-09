import mindspore as ms
from mindspore import nn, ops

def loss_EPM(A, f, f_prime, score):
    '''
        Calculate easy positive mining loss.
        Input: A: sparse adjancy matrix, shape: b, t, t  (n, i, j)
                f: original features, shape: b, t, c
               f_prime: features output by ACGNet, shape: b, t, c
               score: maximum activation score among all the classes, shape: b, t
    '''
    b, t, c = f.shape
    
    f = f.view(b * t, c)
    f_prime = f_prime.view(b * t, c)  
    
    idx = ops.sum(A, dim=2).gt(0).view(-1)
    f = f[idx,:]          # (L, c)
    f_prime = f_prime[idx,:]  # (L, c)
    
    # no such entity as A'_ij > 0
    if f.shape[0] == 0 or f_prime.shape[0] == 0:
        return 0.
    
    score = score.view(-1)
    score = score[idx]
    
    L = f.shape[0]
    
    # f_ite = f.repeat([L, 1])        # (N*M, c)
    # score_ite = score.repeat(L)    # (N*M)
    f_ite = f.tile((L, 1))  # (N*M, c)
    score_ite = score.tile((L,))  # (N*M)
    
    f_prime_chunks = f.repeat_interleave(L, dim=0)  # (M*N, c)
    
    loss = ops.mse_loss(f_ite, f_prime_chunks, reduction='none').sum(-1) * score_ite
    return loss.mean()