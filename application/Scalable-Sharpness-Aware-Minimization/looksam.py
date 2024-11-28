import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor, Parameter, ops, context
import mindspore.common.dtype as mstype
import numpy as np
from typing import Callable, List


class LookSAM(nn.Optimizer):
    def __init__(self,
                 params: List[Parameter],
                 k: int,
                 alpha: float,
                 base_optimizer: nn.Optimizer,
                 criterion: Callable[[Tensor, Tensor], Tensor],
                 rho: float = 0.05,
                 **kwargs):
        """
        LookSAM 算法: https://arxiv.org/pdf/2203.02714.pdf
        """
        super(LookSAM, self).__init__(learning_rate=base_optimizer.learning_rate, parameters=params)
        self.k = k
        self.alpha = alpha
        self.rho = rho
        self.criterion = criterion
        self.base_optimizer = base_optimizer
        self.params = params
        self.global_step = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.eps = Tensor(1e-8, mstype.float32)
        self.gv_list = [Parameter(Tensor(np.zeros_like(p.asnumpy()), mstype.float32), name=f'gv_{i}')
                        for i, p in enumerate(self.params)]
        self.old_params = [Parameter(p.copy(), name=f'old_p_{i}') for i, p in enumerate(self.params)]
        self.old_grads = [Parameter(Tensor(np.zeros_like(p.asnumpy()), mstype.float32), name=f'old_grad_p_{i}')
                          for i, p in enumerate(self.params)]
        self.cast = ops.Cast()

    def construct(self, gradients):
        pass

    def step(self, grads, model, samples, targets):
        t = self.global_step
        t_int = int(t.asnumpy())
        if t_int % self.k == 0:
            grad_norm = self._grad_norm(grads)
            scale = self.rho / (grad_norm + self.eps)

            for idx, (p, g) in enumerate(zip(self.params, grads)):
                self.old_params[idx].set_data(p.copy())
                self.old_grads[idx].set_data(g.copy())
                e_w = g * scale
                p.set_data(p + e_w)

            def forward_fn(samples, targets):
                outputs = model(samples)
                loss = self.criterion(outputs, targets)
                return loss

            perturbed_loss_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=model.trainable_params())
            perturbed_loss, grads_perturbed = perturbed_loss_fn(samples, targets)

            for idx, (old_g, g_s) in enumerate(zip(self.old_grads, grads_perturbed)):
                g_grad_norm = self._normalized(old_g)
                g_s_grad_norm = self._normalized(g_s)
                dot_product = ops.ReduceSum()(g_grad_norm * g_s_grad_norm)
                gv = g_s - ops.norm(g_s, 2) * dot_product * g_grad_norm
                self.gv_list[idx].set_data(gv)
                p = self.params[idx]
                p.set_data(self.old_params[idx])
        else:
            updated_grads = []
            for idx, (g, gv) in enumerate(zip(grads, self.gv_list)):
                norm_g = ops.norm(g, 2)
                norm_gv = ops.norm(gv, 2) + self.eps
                updated_grad = g + self.alpha * (norm_g / norm_gv) * gv
                updated_grads.append(updated_grad)

            grads = tuple(updated_grads)

        self.base_optimizer(grads)
        self.global_step.set_data(self.global_step + 1)

    def _normalized(self, g):
        norm = ops.norm(g, 2) + self.eps
        return g / norm

    def _grad_norm(self, grads):
        norms = [ops.norm(g, 2) for g in grads]
        squared_norms = [n ** 2 for n in norms]
        total_norm = ops.sqrt(ops.ReduceSum()(ops.stack(squared_norms)))
        return total_norm


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    class SimpleModel(nn.Cell):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Dense(10, 1)

        def construct(self, x):
            return self.linear(x)


    data = np.random.randn(100, 10).astype(np.float32)
    targets = np.random.randn(100, 1).astype(np.float32)

    dataset = ds.NumpySlicesDataset({"data": data, "targets": targets}, shuffle=False)
    dataset = dataset.batch(32)

    model = SimpleModel()
    criterion = nn.MSELoss()

    base_optimizer = nn.SGD(params=model.trainable_params(), learning_rate=0.01)
    optimizer = LookSAM(params=model.trainable_params(),
                        k=10,
                        alpha=0.7,
                        base_optimizer=base_optimizer,
                        criterion=criterion,
                        rho=0.05)


    def train_step(samples, targets):
        def forward_fn(samples, targets):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            return loss

        loss_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=model.trainable_params())
        loss, grads = loss_fn(samples, targets)
        optimizer.step(grads, model, samples, targets)
        return loss


    for epoch in range(2):
        for batch_idx, data_batch in enumerate(dataset.create_dict_iterator()):
            samples = data_batch['data']
            targets = data_batch['targets']
            loss = train_step(samples, targets)
            print(f"Epoch {epoch}, Iteration {batch_idx}, Loss: {loss.asnumpy()}")
