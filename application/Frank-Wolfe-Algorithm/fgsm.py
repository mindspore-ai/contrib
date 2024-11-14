import mindspore
from mindspore import nn, ops, Tensor, Parameter, context
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.initializer import initializer
from mindspore import dtype as mstype
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')

context.set_context(mode=context.PYNATIVE_MODE)

def clone_tensor(tensor):
    return Tensor(tensor.asnumpy().copy(), dtype=tensor.dtype)

class FGSM(Optimizer):
    def __init__(self, params, learning_rate, iterT, mergeadam=False, weight_decay=0.0):
        super(FGSM, self).__init__(learning_rate, params)
        self.iterT = iterT
        self.mergeadam = mergeadam
        self.weight_decay = weight_decay
        self.state = {}
        for param in self.parameters:
            self.state[param] = {
                'momentum_buffer': ops.zeros_like(param),
                'data_orig': clone_tensor(param),
                'grad_initial': ops.zeros_like(param)
            }

    def construct(self, gradients, total_batches, first_chunk=None, last_chunk=None):
        if first_chunk is None and last_chunk is None:
            first_iter = (total_batches - 1) % self.iterT == 0
            last_iter = (total_batches - 1) % self.iterT == (self.iterT - 1)
            t = (total_batches - 1) % self.iterT + 1
        else:
            first_iter = first_chunk
            last_iter = last_chunk
            t = (total_batches - 1) % self.iterT + 1

        continue_adam = False
        new_grads = []

        for param, grad in zip(self.parameters, gradients):
            state = self.state[param]
            if grad is None:
                continue

            if first_iter:
                state['data_orig'] = clone_tensor(param)
                state['grad_initial'] = clone_tensor(grad)
                state['momentum_buffer'] = ops.zeros_like(param)

            buf = state['momentum_buffer']
            grad_norm = ops.norm(grad)
            if grad_norm == 0:
                d_p = grad
            else:
                d_p = grad / grad_norm

            buf = buf * (1.0 - 1.0 / t) + (-self.learning_rate / t) * d_p
            param_new = param + buf
            ops.assign(param, param_new)

            grad_new = buf
            new_grads.append(grad_new)

            if last_iter:
                ops.assign(param, state['data_orig'])
                grad_new = grad_new / (-self.learning_rate)
                state['dis'] = ops.reduce_sum(state['grad_initial'] * grad_new)
                continue_adam = True

        return continue_adam, new_grads

class MultipleOptimizer:
    def __init__(self, optimizer_fgsm, optimizer_adam):
        self.optimizer_fgsm = optimizer_fgsm
        self.optimizer_adam = optimizer_adam

    def zero_grad(self):
        pass

    def step(self, total_batches, grads, first_chunk=None, last_chunk=None):
        continue_adam, new_grads = self.optimizer_fgsm(grads, total_batches, first_chunk, last_chunk)
        if continue_adam:
            self.optimizer_adam(new_grads)

if __name__ == '__main__':
    class SimpleModel(nn.Cell):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Dense(10, 50)
            self.fc2 = nn.Dense(50, 1)
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    batch_size = 32
    input_data = Tensor(np.random.randn(batch_size, 10), mindspore.float32)
    target_data = Tensor(np.random.randn(batch_size, 1), mindspore.float32)

    model = SimpleModel()
    criterion = nn.MSELoss()

    fgsm_optimizer = FGSM(model.trainable_params(), learning_rate=1e-3, iterT=10, mergeadam=True)
    adam_optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)

    optimizer = MultipleOptimizer(fgsm_optimizer, adam_optimizer)

    def loss_fn(inputs, targets):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return loss

    grad_fn = mindspore.ops.value_and_grad(loss_fn, None, model.trainable_params())

    num_epochs = 5
    for epoch in range(num_epochs):
        loss, grads = grad_fn(input_data, target_data)
        optimizer.zero_grad()
        optimizer.step(epoch + 1, grads)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.asnumpy():.4f}')