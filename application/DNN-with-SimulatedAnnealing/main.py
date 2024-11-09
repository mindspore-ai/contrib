from neural_network import ResidualBlock, DeeperResidualNN
from simulated_annealing import simulated_annealing

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
import numpy as np

# 定义模型、损失函数和优化器
model = DeeperResidualNN()
criterion = nn.MSELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

num_epochs = 500
for epoch in range(num_epochs):
    model.set_train()  

    x = Tensor(np.random.uniform(-10, 10, (500, 2)), ms.float32)  
    y = Tensor(np.random.uniform(-10, 10, (500, 1)), ms.float32)  

    # 引入小量噪声，增加泛化能力
    noise = Tensor(np.random.normal(0, 0.5, (500, 2)), ms.float32)
    x = x + noise

    def forward_fn(x, y):
        predictions = model(x)
        loss = criterion(predictions, y)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=model.trainable_params())
    loss, grads = grad_fn(x, y)
    optimizer(grads)

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.asnumpy():.4f}')

# 模拟退火参数
initial_solution = Tensor([-3.8, -3.8], ms.float32).expand_dims(0)  
max_temperature = 1000     # 初始温度
min_temperature = 1
cooling_rate = 0.99        # 冷却速率
num_iterations = 1000      # 迭代次数
l = Tensor([-15, -15], ms.float32)
u = Tensor([15, 15], ms.float32)
interval = (l, u)
sigma = 2.0

best_solution, best_value = simulated_annealing(model, initial_solution, max_temperature, min_temperature, cooling_rate, num_iterations, interval, sigma)
print(f"Optimal Solution: {best_value:.4f}, Prediction = {best_solution}")