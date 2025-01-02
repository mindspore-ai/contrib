import mindspore
from mindspore import nn, ops, Parameter
from mindspore.experimental.optim import Adam
from ms_moosvgd import get_gradient
from pymoo.factory import get_problem
from ms_zdt_functions import *
import time
from pymoo.factory import get_performance_indicator
import mindspore.numpy as mnp
import copy
from mindspore import context
context.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target="CPU")
context.set_context(mode=context.PYNATIVE_MODE)
cur_problem = 'zdt3'
run_num = 0
l2_normalize = ops.L2Normalize(axis=0)

if __name__ == '__main__':
    inputs = ops.rand(50, 30)
    x_t = copy.deepcopy(inputs)
    x = Parameter(x_t, name="param", requires_grad=True)
    optimizer = Adam([x], lr=5e-4)
    
    ref_point = get_ref_point(cur_problem)
    hv = get_performance_indicator('hv', ref_point=ref_point)
    iters = 100
    start_time = time.time()
    hv_results = []
    for i in range(iters):
        loss_1, loss_2 = loss_function(x, problem=cur_problem)
        # print(loss_1, loss_2)
        pfront = ops.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], axis=1)
        pfront = pfront.asnumpy()
        hvi = hv.calc(pfront)
        hv_results.append(hvi)

        if i%1000 == 0:
            problem = get_problem(cur_problem)
            x_p = problem.pareto_front()[:, 0]
            y_p = problem.pareto_front()[:, 1]

        loss_1 = loss_1.sum()
        loss_2 = loss_2.sum()
        grad_1, _ = mindspore.grad(loss_function, grad_position=0, has_aux=True)(x)
        grad_2 = mindspore.grad(loss_function, grad_position=0)(x)
        
        # Perforam gradient normalization trick 
        grad_1 = l2_normalize(grad_1)
        grad_2 = l2_normalize(grad_2)
        grad_1 = mindspore.Tensor(grad_1, dtype=mindspore.float32) 
        grad_2 = mindspore.Tensor(grad_2, dtype=mindspore.float32) 
        loss_1 = ops.unsqueeze(loss_1, dim=1)
        loss_2 = ops.unsqueeze(loss_2, dim=1)
        losses = ops.cat([loss_1, loss_2], axis=1)
        x.grad = get_gradient(grad_1, grad_2, x, losses)
        
        optimizer(loss_2)
        x.data = ops.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)

    print(i, 'time:', time.time()-start_time, 'hv:', hvi, loss_1.sum().numpy(), loss_2.sum().numpy())
