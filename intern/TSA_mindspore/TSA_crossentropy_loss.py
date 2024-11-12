import mindspore
from mindspore import nn, ops, Tensor, Parameter
import numpy as np


class TSA_crossEntropy:
    def __init__(self, num_steps, num_classes, alpha, temperature):
        self.loss_function = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.num_steps = num_steps
        self.current_step = 0
        self.num_classes = num_classes
        self.alpha = alpha
        self.temperature = temperature

    def thresh(self):
        ratio = self.current_step / self.num_steps
        exp_value = ops.exp(Tensor([(ratio - 1) * self.alpha], mindspore.float32))
        thresh = exp_value * (1 - 1 / self.num_classes) + 1 / self.num_classes
        return thresh

    def step(self):
        self.current_step += 1

    def get_mask(self, logits, targets):
        thresh = self.thresh()
        softmax = ops.Softmax(axis=1)(logits)
        max_probs = ops.ReduceMax(keep_dims=False)(softmax, axis=1)
        pred = ops.Argmax(axis=1)(softmax)

        wrong_pred = ops.Abs()(pred - targets) > 0
        mask = max_probs < thresh
        mask = ops.logical_or(mask, wrong_pred)
        mask = mask.astype(mindspore.float32)
        return mask

    def __call__(self, logits, targets):
        logits = logits / self.temperature
        mask = self.get_mask(logits, targets)
        loss_value = self.loss_function(logits, targets)
        loss_value = loss_value * mask
        loss_sum = ops.ReduceSum()(loss_value)
        mask_sum = ops.ReduceSum()(mask)
        loss_value = loss_sum / ops.maximum(mask_sum, Tensor(1.0, mindspore.float32))
        self.step()  
        return loss_value


class SimpleModel(nn.Cell):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.weight = Parameter(Tensor(np.zeros((2, 2), dtype=np.float32)), name='weight')
        self.bias = Parameter(Tensor(np.zeros((1, 2), dtype=np.float32)), name='bias')

    def construct(self, x):
        logits = ops.matmul(x, self.weight) + self.bias
        return logits


loss = TSA_crossEntropy(num_steps=10000, num_classes=2, alpha=5, temperature=5)

inputs = Tensor(np.array([[11,10],[200,151],[501,400],[6,7],[7,8],[8,9],[9,10],[10,11],
                          [11,12],[12,13],[1,10],[9,20],[9,100],[100,101],[1111,1112]], dtype=np.float32))
targets = Tensor(np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]), dtype=mindspore.int32)

model = SimpleModel()

for i in range(10000):
   
    optimizer = nn.SGD(params=model.trainable_params(), learning_rate=0.01, momentum=0.9)
    def forward_fn(x, y):
        logits = model(x)
        loss_value = loss(logits, y)
        return loss_value

    grad_fn = mindspore.ops.value_and_grad(forward_fn, grad_position=None, weights=model.trainable_params())
    loss_value, grads = grad_fn(inputs, targets)
    optimizer(grads)
   


test_inputs = Tensor(np.array([[200,151],[501,400],[9,8],[99,95],[100,100000],[99,60]], dtype=np.float32))
logits = model(test_inputs)
softmax = ops.Softmax(axis=1)(logits)
print('with TSA training')
print('weights:', model.weight.asnumpy())
print('bias:', model.bias.asnumpy())
print('output:', softmax.asnumpy())

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')


model = SimpleModel()

for i in range(10000):
    optimizer = nn.SGD(params=model.trainable_params(), learning_rate=0.01, momentum=0.9)
    def forward_fn(x, y):
        logits = model(x)
        loss_value = loss_fn(logits, y)
        return loss_value

    grad_fn = mindspore.ops.value_and_grad(forward_fn, grad_position=None, weights=model.trainable_params())
    loss_value, grads = grad_fn(inputs, targets)
    optimizer(grads)


logits = model(test_inputs)
softmax = ops.Softmax(axis=1)(logits)
print('without TSA training')
print('weights:', model.weight.asnumpy())
print('bias:', model.bias.asnumpy())
print('output:', softmax.asnumpy())
