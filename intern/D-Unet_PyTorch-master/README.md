# D-Unet - MindSpore

This code is a mindspore implementation of D-Unet 

## Usage

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeUniform, Zero, initializer

from DUnet_parts_ms import *
from loss_ms import *

BATCH_SIZE = 4
x_2d = ms.Tensor(shape=(BATCH_SIZE, 4, H, W), dtype=ms.float32, init=ms.common.initializer.Normal()) #2D输入 (例如：多通道MRI切片)
#x_3d = ms.Tensor(shape=(BATCH_SIZE, 1, 8, H, W), dtype=ms.float32, init=ms.common.initializer.Normal())  #3D输入 (可选，如果有真实的3D数据)

model = DUnet_ms(in_channels_2d=4, in_channels_3d=1)

output = model(x_2d)  # 自动生成3D输入
# output = model(x_2d, x_3d)  # 如果有真实3D输入
```

- According to the Reference paper input size must be (4, 192, 192) and output size must be (1, 192, 192)

## Project Structure
- DUnet_ms.py
    - DUnet_parts_ms.py
    - loss_ms.py

## Reference

[1] Yongjin Zhou et al., D-UNet: a dimension-fusion U shape network for chronic stroke lesion segmentation ([ arXiv:1908.05104](https://arxiv.org/abs/1908.05104) [eess.IV] ), 2019 Aug

[2] SZUHvern github source code implemented with keras (https://github.com/SZUHvern/D-UNet)