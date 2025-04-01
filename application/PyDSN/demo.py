import numpy as np
from mindspore import Tensor

from loss_mindspore import entropy_loss, cyclostationary, kurtosis_loss, intelligent_spectrogram_loss, Intelligent_spectrogram_loss, Calculate_renyi_entropy


x = Tensor(np.random.randn(4, 64, 128).astype(np.float32))


# 1. 计算熵损失
loss_entropy = entropy_loss(x)
print("Entropy Loss:", loss_entropy.asnumpy())

# 2. 计算循环平稳性指标
cyc_value = cyclostationary(x)
print("Cyclostationary Value:", cyc_value.asnumpy().real)

# 3. 计算峰度损失
loss_kurtosis = kurtosis_loss(x)
print("Kurtosis Loss:", loss_kurtosis.asnumpy())

# 4. 计算智能频谱损失（函数实现）
loss_intelligent_spec = intelligent_spectrogram_loss(x)
print("Intelligent Spectrogram Loss (function):", loss_intelligent_spec.asnumpy())

# 5.计算智能频谱损失（nn.Cell 实现）
loss_module = Intelligent_spectrogram_loss()
loss_module_value = loss_module(x)
print("Intelligent Spectrogram Loss (module):", loss_module_value.asnumpy())

# 6. 计算 Renyi 熵
renyi_module = Calculate_renyi_entropy(bins=50, alpha=2)
time_freq_map = x[0]
renyi_entropy_value = renyi_module([time_freq_map])
print("Renyi Entropy:", renyi_entropy_value.asnumpy())
