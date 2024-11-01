import mindspore
from mindspore import nn, ops
from mindspore.experimental import optim

class DGC_SGD(optim.SGD):
    def __init__(self, params, learning_rate=0.1, momentum=0, dampening=0.0, weight_decay=0.0, nesterov=False, compress_ratio=0.01):
        super(DGC_SGD, self).__init__(params, learning_rate, momentum, dampening,
                                      weight_decay, nesterov)
        self.momentum = momentum
        self.grads_accumulator = {}
        self.nesterov = nesterov
        self.compress_ratio = compress_ratio

    
    def construct(self, gradients):
        params = self.parameters
        for param, grad in zip(params, gradients):
                # print('param', param)
                if grad is None:
                    continue
                # Add gradient to the accumulator
                if param in self.grads_accumulator:
                    self.grads_accumulator[param] += grad
                else:
                    self.grads_accumulator[param] = grad.copy()

                # Apply DGC compression
                if self.compress_ratio > 0:
                    self.compress_gradients(param)

                # Update the parameters using SGD
                if self.momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = ops.zeros_like(param)
                        buf = ops.mul(buf, self.momentum)
                        buf = ops.add(buf, grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf = ops.mul(buf, self.momentum)
                        buf = ops.add(buf, grad)
                    if self.nesterov:
                        temp = ops.add(grad, self.momentum)
                        grad = ops.add(temp, buf)    
                    else:
                        grad = buf

                    # Apply momentum correction for DGC
                    if param in self.momentum_correction:
                        momentum_correction = self.momentum_correction[param]
                        grad = ops.add(grad, -momentum_correction)

                    # Update momentum correction for DGC
                    momentum_correction = grad.clone()
                    momentum_correction = ops.mul(momentum_correction, self.momentum)
                    self.momentum_correction[param] = momentum_correction
                group = self.param_groups
                lr = - group[0]['lr']
                param = ops.add(param, lr * grad)

        return params

    def compress_gradients(self, param):
        grad = self.grads_accumulator[param]
        numel = grad.numel()

        # Determine the threshold value for sparsification
        k = int(numel * self.compress_ratio)
        if k == 0:
            return

        # Sparsify gradients
        _, indices = ops.topk(grad.abs().view(-1), k)
        mask = ops.zeros_like(grad)
        mask.view(-1).index_fill_(0, indices, 1)
        grad.mul_(mask)

        # Quantize gradients
        grad.div_(mask.sum())
        grad_rounded = grad.round()
        grad_quantized = grad_rounded / grad.numel()

        # Update the gradients accumulator with compressed gradients
        self.grads_accumulator[param] = grad_quantized

        # Clear the accumulator if all workers have updated the gradients
        if mindspore.communication.get_rank() == 0:
            self.grads_accumulator[param] = ops.zeros_like(grad)

class SimpleNN(nn.Cell):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Dense(10, 5)
        self.fc2 = nn.Dense(5, 1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
loss_function = nn.MSELoss()

def forward_fn(data, label):
    output = model(data)  # 前向传播
    loss = loss_function(output, label)  # 计算损失
    return loss, output  # 返回损失和输出

def main():
    # 初始化模型和优化器
    optimizer = DGC_SGD(model.trainable_params(), learning_rate=0.01)
    inputs = mindspore.ops.rand(32, 10)  # 32个样本，每个样本10个特征
    targets = mindspore.ops.rand(32, 1)   # 32个目标值
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # 计算损失和梯度
    (loss, _), grads = grad_fn(inputs, targets)
    print(f'grads: {grads}')
    optimizer.construct(grads)  # 更新参数
    print(f'Loss: {loss.asnumpy()}')

if __name__ == "__main__":
    main()
