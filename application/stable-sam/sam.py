from collections import defaultdict
import mindspore as ms
from mindspore import ops
from mindspore.experimental.optim import Optimizer


class SAM(Optimizer):
    def __init__(self, base_optimizer: Optimizer, rho=0.05, adaptive=False, stable=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, stable=stable)
        if stable:
            defaults.update({'exact_gradient_norm': 1.0, 'surrogate_gradient_norm': 1.0})

        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        super(SAM, self).__init__(base_optimizer.parameters, self.defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.base_optimizer.param_groups:
                group.setdefault(name, default)

    def first_step(self, gradients):
        grad_norm = self._grad_norm(gradients)
        for group_id, group in enumerate(self.param_groups):
            id = self.group_start_id[group_id]
            if group['stable']: group['exact_gradient_norm'] = grad_norm.item()
            scale = group["rho"] / (grad_norm + 1e-12)

            for i, p in enumerate(group['params']):
                if gradients[id+i] is None: continue
                grad = gradients[id+i]
                self.state[p]["old_p"] = p.clone()
                e_w = (ops.pow(p, 2) if group["adaptive"] else 1.0) * grad * scale.to(p.dtype)
                p.set_data(p.add(e_w))  # climb to the local maximum "w + e(w)"

    def second_step(self, gradients):
        if self.param_groups[0]['stable']: grad_norm = self._grad_norm(gradients)
        for group_id, group in enumerate(self.param_groups):
            id = self.group_start_id[group_id]
            if group['stable']:
                group['surrogate_gradient_norm'] = grad_norm.item()
                scale = group['exact_gradient_norm'] / (group['surrogate_gradient_norm'] + 1.0e-12)
            for i, p in enumerate(group['params']):
                if gradients[id+i] is None: continue
                grad = gradients[id+i]
                p.set_data(self.state[p]["old_p"])  # get back to "w" from "w + e(w)"
                if group['stable']: ops.assign(gradients[id+i], grad.mul(scale))  # downscale the gradient magnitude to the same as the exact gradient
                # if group['stable']: ops.assign(gradients[id+i], grad.mul(min(1, scale))  # e.g. explicitly truncate the upper bound to further enhance stability
        self.base_optimizer(gradients)  # do the actual "sharpness-aware" update

    def construct(self, gradients, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"

        self.first_step(gradients)     
        _, grads = closure()
        self.second_step(grads)

    def _grad_norm(self, gradients):
        norm = ops.norm(
                    ops.stack([
                        ((ops.abs(p) if group["adaptive"] else 1.0) * gradients[self.group_start_id[group_id]+i]).norm(ord=2)
                        for group_id, group in enumerate(self.param_groups) for i, p in enumerate(group['params'])
                        if gradients[self.group_start_id[group_id]+i] is not None
                    ]),
                    ord=2
               )
        return norm

    def load_state_dict(self, state_dict):
        ms.load_param_into_net(self, state_dict)
        self.param_groups = self.base_optimizer.param_groups


if __name__ == '__main__':
    import mindspore.nn as nn
    from mindspore.experimental import optim
    import mindspore.dataset as ds
    import mindspore.dataset.vision as vision
    import matplotlib.pyplot as plt
    
    class Model(nn.Cell):
        def __init__(self):
            super(Model, self).__init__()
            self.fc = nn.Dense(28 * 28, 10)

        def construct(self, x):
            return self.fc(x.view(-1, 28 * 28))

    def create_dataset():
        # from download import download
        # url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
        #     "notebook/datasets/MNIST_Data.zip"
        # path = download(url, "./data", kind="zip", replace=True)
        
        mnist_train = ds.MnistDataset("./data/MNIST_Data/train")
        transform = [vision.Rescale(1.0 / 255.0, 0.0), vision.Normalize(mean=(0.5,), std=(0.5,))]
        mnist_train = mnist_train.map(operations=transform, input_columns="image")
        mnist_train = mnist_train.batch(64, drop_remainder=True)
        return mnist_train

    train_loader = create_dataset()

    def train(model, train_loader, optimizer, grad_fn):
        model.set_train()
        losses = []
        for data in train_loader.create_dict_iterator():
            data_image, data_label = data['image'], data['label'].to(ms.int32)
            (loss, _), grads = grad_fn(data_image, data_label)
            
            def closure():
                (loss, _), grads = grad_fn(data_image, data_label)
                losses.append(loss)
                return loss, grads

            optimizer(grads, closure)
            
        plt.plot(losses)
        plt.show()

    criterion = nn.CrossEntropyLoss()

    def forward_fn(data, label):
        output = model(data)
        loss = criterion(output, label)
        return loss, output

    model = Model()
    base_optimizer = optim.SGD(params=model.trainable_params(), lr=0.1)
    sam_optimizer = SAM(base_optimizer)
    grad_fn = ms.value_and_grad(forward_fn, None, sam_optimizer.parameters, has_aux=True)
    train(model, train_loader, sam_optimizer, grad_fn)
