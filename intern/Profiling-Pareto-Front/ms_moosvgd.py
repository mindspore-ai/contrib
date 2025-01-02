import math
import mindspore
from mindspore import ops
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE)

def solve_min_norm_2_loss(grad_1, grad_2):
    a = ops.mul(grad_1, grad_1)
    a = mindspore.Tensor(a, dtype=mindspore.float32)
    v1v1 = ops.sum(grad_1*grad_1, dim=1)
    v2v2 = ops.sum(grad_2*grad_2, dim=1)
    v1v2 = ops.sum(grad_1*grad_2, dim=1)
    gamma = ops.zeros_like(v1v1)
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
    gamma[v1v2>=v1v1] = 0.999
    gamma[v1v2>=v2v2] = 0.001
    gamma = gamma.view(-1, 1)
    g_w = gamma.repeat(1, grad_1.shape[1])*grad_1 + (1.-gamma.repeat(1, grad_2.shape[1]))*grad_2

    return g_w

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (ops.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def kernel_functional_rbf(losses):
    n = losses.shape[0]
    pairwise_distance = ops.norm(losses[:, None] - losses, dim=2).pow(2)
    h = median(pairwise_distance) / math.log(n)
    kernel_matrix = ops.exp(-pairwise_distance / 5e-6*h) #5e-6 for zdt1,2,3 (no bracket)
    return kernel_matrix

def get_gradient(grad_1, grad_2, inputs, losses):
    n = 50
    print('n',n)
    #inputs = inputs.detach().requires_grad_(True)
    g_w = solve_min_norm_2_loss(grad_1, grad_2)
    ### g_w (100, x_dim)
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_functional_rbf(losses)
    kernel_grad = -0.5 * ops.grad(kernel.sum(), inputs, allow_unused=True)[0]

    gradient = (kernel.mm(g_w) - kernel_grad) / n

    return gradient
