import mindspore
from mindspore import nn, ops, Tensor, ParameterTuple
from mindspore import context
import numpy as np
import random
from itertools import accumulate

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.freeze = nn.Dense(10, 10)  # 冻结的层
        self.base1 = nn.Dense(10, 10)
        self.task1 = nn.Dense(10, 2)
        self.task2 = nn.Dense(10, 2)
        self.relu = ops.ReLU()

        # 冻结参数
        for p in self.freeze.get_parameters():
            p.requires_grad = False

    def construct(self, x):
        x = self.relu(self.freeze(x))
        x = self.relu(self.base1(x))
        t1 = self.task1(x)
        t2 = self.task2(x)
        return (t1, t2)

def PCGrad_backward(net, optimizer, X, y, loss_layer):
    params = net.trainable_params()
    grad_numel = [int(np.prod(p.shape)) for p in params]
    num_tasks = len(y)
    losses = []

    indices = [0] + [int(v) for v in accumulate(grad_numel)]

    grads_task = []

    # 计算每个任务的梯度
    for i in range(num_tasks):
        def loss_fn():
            outputs = net(X)
            loss = loss_layer(outputs[i], y[i])
            return loss

        grad_fn = mindspore.ops.value_and_grad(loss_fn, grad_position=None, weights=params)
        loss, grads = grad_fn()
        losses.append(loss)

        # 将梯度展开并拼接
        flat_grads = []
        for grad in grads:
            flat_grads.append(grad.view(-1))
        flat_grads = ops.concat(flat_grads)
        grads_task.append(flat_grads)

    # 打乱梯度顺序
    random.shuffle(grads_task)

    # 梯度投影
    grads_task = ops.stack(grads_task, axis=0)
    proj_grad = grads_task.copy()

    def _proj_grad(grad_task):
        for k in range(num_tasks):
            inner_product = ops.ReduceSum()(grad_task * grads_task[k])
            grad_norm = ops.ReduceSum()(grads_task[k] * grads_task[k]) + 1e-12
            proj_direction = inner_product / grad_norm
            min_proj = ops.Minimum()(proj_direction, Tensor(0.0, mindspore.float32))
            grad_task = grad_task - min_proj * grads_task[k]
        return grad_task

    proj_grad_list = []
    for g in proj_grad:
        proj_grad_list.append(_proj_grad(g))

    proj_grad = ops.ReduceSum()(ops.stack(proj_grad_list), axis=0)

    # 将投影后的梯度分配给对应的参数
    grads = []
    for param, start_idx, end_idx in zip(params, indices[:-1], indices[1:]):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        grad = proj_grad[start_idx:end_idx].view(param.shape)
        grads.append(grad)

    # 将参数列表和梯度列表转换为元组
    params = ParameterTuple(params)
    grads = tuple(grads)

    # 调用优化器更新参数
    optimizer(grads)

    return losses

if __name__ == '__main__':
    net = Net()
    net.set_train()
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    loss_layer = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    num_task = 2
    num_iterations = 50000 # 为了演示，减少迭代次数

    for it in range(num_iterations):
        X = Tensor(np.random.rand(20, 10).astype(np.float32)) - Tensor(
            np.concatenate([np.zeros((10, 10)), np.ones((10, 10))], axis=0).astype(np.float32))
        y = [
            Tensor(np.concatenate([np.zeros(10,), np.ones(10,)]).astype(np.int32)),
            Tensor(np.concatenate([np.ones(10,), np.zeros(10,)]).astype(np.int32))
        ]

        losses = PCGrad_backward(net, optimizer, X, y, loss_layer)
        # 不需要调用 optimizer.step()，因为已经在 PCGrad_backward 中更新了参数

        if it % 100 == 0:
            total_loss = sum([loss.asnumpy() for loss in losses])
            print(f"iter {it} total loss: {total_loss}")
