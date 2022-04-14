# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
An implementation of UKF Algorithm.
"""
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from mindspore import Tensor

# Algorithm1 UKF(论文中无迹卡尔曼滤波算法实现)
# 无迹卡尔曼滤波基于传统卡尔曼滤波进行了改进，状态变量按非线性方式变化
# 假设：观测变量和测量变量服从高斯分布
# 定义数据数组，预测数组和噪声数组
_ls_x_true = []
_ls_y = []
_ls_x_noise = []

# 数据规模定义，这里可根据实际数据集规模进行选择
DATA_ROWS = 360
X_TRUE = 100

# 采用一组随机采样点来描述随机变量的高斯分布
for i in range(DATA_ROWS):
    _ls_x_true.append(X_TRUE)
    # 生成随机采样点
    noise_s = Tensor(np.random.normal(loc=0, scale=2.5))
    x_with_noise = X_TRUE + noise_s
    _ls_x_noise.append(x_with_noise)
    _ls_y.append(sqrt(x_with_noise))

X_SI = _ls_y[0] ** 2
_P = 0.1
_Q = 0.05
R = 2

# UKF算法-初始化
# 设定均值权值w_m0和方差权值w_c/w_c0，以便近似非线性函数的后验
# 均值和方差
K_VAL = 0.2
ALPHA_VAL = 0.9
L = 1
BETA_VAL = 2
lamb = ALPHA_VAL * ALPHA_VAL * (L + K_VAL) - L
w_c = 0.5 / (ALPHA_VAL * ALPHA_VAL * (L + K_VAL))
w_m0 = lamb / (ALPHA_VAL * ALPHA_VAL * (L + K_VAL))
w_c0 = lamb / (ALPHA_VAL * ALPHA_VAL * (L + K_VAL)) + 1 - ALPHA_VAL ** 2 + BETA_VAL

ls_x_hat = []
LS_P = []

# UKF算法-核心部分：预测、更新和权值迭代
for i in range(DATA_ROWS):
    # Prediction(式16、18）
    _P = _P + _Q

    # Update(式23、25）
    x0 = X_SI
    x1 = X_SI + sqrt((L + lamb) * _P)
    x2 = X_SI - sqrt((L + lamb) * _P)

    y0 = sqrt(x0)
    y1 = sqrt(x1)
    y2 = sqrt(x2)

    # Validation(式20、21）
    y_hat = w_m0 * y0 + w_c * y1 + w_c * y2
    y = _ls_y[i]
    s_hat = w_c0 * (y_hat - y0) * (y_hat - y0) +\
            w_c * (y_hat - y1) * (y_hat - y1) + w_c * (y_hat - y2) * (y_hat - y2)
    s_hat = s_hat + R

    C_sz = (x0 - X_SI) * (y0 - y_hat) * w_c0 + (x1 - X_SI) * \
           (y1 - y_hat) * w_c + (x2 - X_SI) * (y2 - y_hat) * w_c

    K_k = C_sz / s_hat
    X_SI = X_SI + K_k * (y - y_hat)

# 结果保存到ls_x_hat中，便于下面进行图形绘制
    _P = _P - K_k * s_hat * K_k
    LS_P.append(_P)

    ls_x_hat.append(X_SI)

# 绘制预测值(hat)，真实值(true)和噪声(noise)
plt.plot(ls_x_hat, label='x_hat')
plt.plot(_ls_x_true, label='x_true')
plt.plot(_ls_x_noise, label='x_noise')
plt.legend()
plt.grid(True)
plt.grid(linestyle='--')
plt.show()

# 结果评估
plt.plot(LS_P, label='P')
plt.legend()
plt.grid(True)
plt.grid(linestyle='--')
plt.show()
