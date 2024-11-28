import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from timeit import default_timer as timer
import os
from TrustRegion import ConjugateGradientTrustRegion
import sys

# 计算copt
class COPT_Cell(nn.Cell):
    def __init__(self, x0):
        super(COPT_Cell, self).__init__()
        self.x0 = mindspore.Parameter(x0, requires_grad=True)

    def construct(self,epsilon):
        dimen = self.x0.shape[1] // 2
        norm = ops.L2Normalize(axis=1)(self.x0)
        x3 = self.x0 / norm
        x1 = x3[:, :dimen]
        x2 = x3[:, dimen:]
        xxt1 = ops.MatMul()(x1, x1.T)
        xxt2 = ops.MatMul()(x2, x2.T)
        xxt3 = ops.MatMul()(x2, x1.T)
        xxt4 = ops.MatMul()(x1, x2.T)
        xxt = (xxt1 + xxt2) ** 2 + (xxt3 - xxt4) ** 2
        xxt = ops.triu(xxt, 1)
        s, _= ops.max(xxt)
        expxxt = ops.Exp()((xxt - s) / epsilon)
        
        u = ops.triu(expxxt, 1).sum()
        f = s + epsilon * ops.Log()(u)
        return f

# 计算coh
def coh(x):
    N = x.shape[0]
    dimen = x.shape[1] // 2
    norm = ops.L2Normalize(axis=1)(x)
    x = x / norm
    x1 = x[:, :dimen]
    x2 = x[:, dimen:]
    xxt1 = ops.MatMul()(x1, x1.T)
    xxt2 = ops.MatMul()(x2, x2.T)
    xxt3 = ops.MatMul()(x2, x1.T)
    xxt4 = ops.MatMul()(x1, x2.T)
    xxt = (xxt1 + xxt2) ** 2 + (xxt3 - xxt4) ** 2
    return ops.Sqrt()(ops.max(ops.triu(xxt, 1))[0])

# 最小化相干性
def min_coh_mindspore(N, d, epsilon, x0):

    coptcell = COPT_Cell(x0)
    optimizer = ConjugateGradientTrustRegion(coptcell.trainable_params())
    grad_func = ops.value_and_grad(coptcell,grad_position=None,weights=coptcell.trainable_params())

    loss1 = 0
    start = timer()
    for l in range(20 * N):

        loss,grad = grad_func(epsilon)

        optimizer(grad)
        
        if abs(loss - loss1) < 1e-10 and (l + 1) % 5 == 0:
            break
        if ops.norm(grad[0]).item() < gtol:
            break
        if ops.norm(optimizer.step_size, dim=-1).lt(gtol).all():
            break
        loss1 = loss
        if (l + 1) % 20 == 0:
            print(f'MS iter:{l+1}, loss={loss1.item()}')

    end = timer()
    elapsed = end - start
    print(f'Stage {-int(np.floor(np.log(epsilon)/np.log(10)))} Complete: (d={d}, N={N})')
    print(f'Coherence: {coh(coptcell.x0)} ')
    return coptcell.x0, elapsed

# 保存运行结果
def save_run_mindspore(d, N, coh, x, eps, trial, elapsed):
    os.makedirs('optc-64e6t-d-{0}'.format(d), exist_ok=True)
    os.makedirs('optc-64e6t-d-{0}/optc-d-{0}-n-{1}'.format(d, N), exist_ok=True)
    np.savetxt('optc-64e6t-d-{0}/optc-d-{0}-n-{1}/optc-d-{0}-n-{1}-coh-{2}-eps-{3}-trial-{4}-time-{5}.txt'.format(d, N, coh, eps, trial, elapsed), x.asnumpy(), delimiter=' ', newline='\n')

# 多次试验
def run_trials_mindspore(d1, d2, n1, n2, eps_pows, n_trials):
    for d in range(d1, d2):
        for N in range(n1, n2):
            for trial in  range(1, n_trials + 1):
                print('')
                print('')
                x0 = mindspore.Tensor(np.random.randn(N, 2*d), dtype=mindspore.float32)
                for epsilon, eps_pow in zip(10.**(-eps_pows), eps_pows):
                    x0, elapsed = min_coh_mindspore(N, d, epsilon, x0)
                    save_run_mindspore(d, N, coh(x0), x0, eps_pow, trial, elapsed)






if len(sys.argv) <5:
 print('Error: trstmi takes 4 arguments or 8. The full input parameters (in order) are dim1 dim2 num1 num2 trials tol proc. "dim1" is the lower bound on dimension, "dim2" is the upper. "num1" and "num2" serve a similar purpose except give the number of points to optimize over. "trials" gives the number of different starting random initializations to optimize over. "tol" is the stopping gradient tolerance. "proc" takes arguments cpu/gpu. "verbose" takes arguments 0/1.')
 exit(1)
 
dim1=int(float(eval(sys.argv[1])))
dim2=int(float(eval(sys.argv[2])))
num1=int(float(eval(sys.argv[3])))
num2=int(float(eval(sys.argv[4])))


if len(sys.argv) >5 and len(sys.argv) <9:
 print('Error: trstmi takes 4 arguments or 8. The full input parameters (in order) are dim1 dim2 num1 num2 trials tol proc. "dim1" is the lower bound on dimension, "dim2" is the upper. "num1" and "num2" serve a similar purpose except give the number of points to optimize over. "trials" gives the number of different starting random initializations to optimize over. "tol" is the stopping gradient tolerance. "proc" takes arguments cpu/gpu. "verbose" takes arguments 0/1.')
 exit(1)


if len(sys.argv) > 5:
 trials=int(float(eval(sys.argv[5])))
 tol=float(eval(sys.argv[6]))
 proc=sys.argv[7]
 verbose=sys.argv[8]
else:
 trials=100
 tol=1e-6
 verbose=0


gtol=tol
run_trials_mindspore(dim1,dim2,num1,num2, mindspore.ops.arange(1,16, dtype=mindspore.int32), trials)