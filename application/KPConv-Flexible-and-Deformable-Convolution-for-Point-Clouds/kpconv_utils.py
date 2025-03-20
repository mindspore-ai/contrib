import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

def linearKernel(yi, xi, sigma):
    return ops.ReLU()(1 - ops.norm(yi - xi, dim=-1) / sigma)

def repulsive_loss(xk, xl):
    # Specify the axis for LpNorm
    lp_norm = ops.LpNorm(axis=-1, p=2)
    return 1 / lp_norm(ops.Sub()(xk, xl))

def attractive_loss(xk):
    # Specify the axis for LpNorm
    lp_norm = ops.LpNorm(axis=-1, p=2)
    square = ops.Square()
    return square(lp_norm(xk))

def total_loss(kerns):
    loss = mindspore.Tensor([[0]], dtype=mindspore.float32)
    for i in range(kerns.shape[0]):
        att_loss = attractive_loss(kerns[i])
        loss = loss + att_loss
        for j in range(kerns.shape[0]):
            if j != i:
                rep_loss = repulsive_loss(kerns[i], kerns[j])
                loss = loss + rep_loss
    return loss

def initializeRigitKernels(numKernels, numItterations=100):
    # solving an optimization problem to position the kernels
    # setting learning rate
    lr = 0.1
    # positioning a point at the origin
    origin = mindspore.Tensor([[0, 0, 0]], dtype=mindspore.float32)
    # positioning the other kernels at random points
    kernels = mindspore.Tensor(mindspore.numpy.rand(numKernels - 1, 3), dtype=mindspore.float32)
    kernels.requires_grad = True

    # Trainning with the original learning rate for the 90% percent of the itterations
    for i in range(numItterations):
        # We want the points to be as far as possible inside a given sphere
        kerns = ops.Concat(axis=0)([origin, kernels])
        # computing the loss
        def forward_fn(kerns):
            return total_loss(kerns)
        grad_fn = ops.grad(forward_fn)
        grads = grad_fn(kerns)
        # updating the kernel positions
        kernels = kernels - lr * grads[1:]
        # reseting the gradient
        # In MindSpore, we don't need to manually reset the gradient like PyTorch
    # The kernels no more require grad computation for the rigit kernel
    kernels.requires_grad = False
    return ops.Concat(axis=0)([origin, kernels])
