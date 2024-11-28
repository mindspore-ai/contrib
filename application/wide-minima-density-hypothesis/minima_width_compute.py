import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import numpy as np
import copy

import mindspore.dataset as ds


class ComputeKeskarSharpness:
    def __init__(self, final_model, optimizer, criterion, trainloader, epsilon=1e-4, lr=0.001, max_steps=1000):
        self.net = final_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.epsilon = epsilon
        self.lr = lr
        self.max_steps = max_steps

        self.optimizer.learning_rate = Parameter(Tensor(self.lr, ms.float32), name="lr")

    def compute_loss(self):
        loss = 0
        self.net.set_train(False)
        train_loss = 0
        total = 0
        for data in self.trainloader.create_dict_iterator():
            inputs, targets = data['input'], data['label']
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets).asnumpy()

            train_loss += loss * targets.shape[0]
            total += targets.shape[0]

        return train_loss / total

    def compute_sharpness(self):
        stop = False
        num_steps = 0
        final_loss = 0
 
        self.net.set_train(False)
        # init loss  --> l_0
        init_loss = self.compute_loss()

        # min and max parameter values
        init_sd = copy.deepcopy(dict(self.net.parameters_dict()))
        max_sd = {k: self.epsilon * (np.abs(v.asnumpy()) + 1.0) for k, v in init_sd.items()}

        def proj(v, k):
            return np.maximum(init_sd[k].asnumpy() - max_sd[k],
                              np.minimum(v.asnumpy(), init_sd[k].asnumpy() + max_sd[k]))

        while not stop:
            for data in self.trainloader.create_dict_iterator():
                x, y = data['input'], data['label']
                self.net.set_train(True)

                # maximize loss
                def forward_fn(inputs, labels):
                    outputs = self.net(inputs)
                    loss = -1 * self.criterion(outputs, labels)
                    return loss

                grad_fn = ms.value_and_grad(forward_fn, grad_position=None, weights=self.net.trainable_params())
                loss, grads = grad_fn(x, y)

                self.optimizer(grads)

                for param in self.net.trainable_params():
                    proj_value = proj(param, param.name)
                    param.set_data(Tensor(proj_value, ms.float32))

                num_steps += 1
                if num_steps == self.max_steps:
                    self.net.set_train(False)
                    final_loss = self.compute_loss()
                    stop = True
                    break

        sharpness = ((final_loss - init_loss) * 100) / (1 + init_loss)
        print("sharpness = ", sharpness)
        return sharpness



def main():

    net = nn.Dense(10, 2)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=0.001)
    criterion = nn.MSELoss()

    data = {
        "input": Tensor(np.random.randn(100, 10), ms.float32),
        "label": Tensor(np.random.randn(100, 2), ms.float32),
    }
    trainloader = ds.NumpySlicesDataset(data, shuffle=True).batch(10)

    sharpness_computer = ComputeKeskarSharpness(net, optimizer, criterion, trainloader)

    sharpness = sharpness_computer.compute_sharpness()
    print("Final sharpness: ", sharpness)


if __name__ == "__main__":
    main()