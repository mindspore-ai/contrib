# Convolutional Gated Recurrent Unit (ConvGRU) Implementation

This repository provides implementations of the Convolutional Gated Recurrent Unit (ConvGRU) in both PyTorch and MindSpore frameworks. The ConvGRU is implemented as described in [Ballas *et. al.* 2015: Delving Deeper into Convolutional Networks for Learning Video Representations](https://arxiv.org/abs/1511.06432).

## Implementations

### PyTorch Implementation
The original implementation in PyTorch includes:
- `convgru.py`: Contains the PyTorch implementation of `ConvGRUCell` and `ConvGRU`
- `convgru_model.py`: Example usage of the PyTorch implementation

### MindSpore Implementation
The MindSpore implementation (based on the PyTorch version) includes:
- `mindspore_convgru.py`: Contains the MindSpore implementation of `ConvGRUCell` and `ConvGRU`
- `mindspore_convgru_model.py`: Example usage of the MindSpore implementation

## Requirements
- For PyTorch version: PyTorch
- For MindSpore version: MindSpore 2.3.0

Install MindSpore dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Both implementations preserve spatial dimensions across cells, only altering depth. The main differences are in the framework-specific syntax and operations.

### PyTorch Example:
```python
from convgru import ConvGRU
import torch

model = ConvGRU(input_size=8, hidden_sizes=[32, 64, 16],
                kernel_sizes=[3, 5, 3], n_layers=3)
x = torch.FloatTensor(1, 8, 64, 64)
output = model(x)
```

### MindSpore Example:
```python
from mindspore_convgru import ConvGRU
import mindspore as ms

model = ConvGRU(input_size=8, hidden_sizes=[32, 64, 16],
                kernel_sizes=[3, 5, 3], n_layers=3)
x = ms.numpy.randn(1, 8, 64, 64)
output = model(x)
```

## Key Differences
- PyTorch uses `forward()` while MindSpore uses `construct()`
- Different initialization methods (PyTorch's init vs MindSpore's initializers)
- Framework-specific tensor operations and neural network modules
- MindSpore requires context setup for execution mode

## Reference
[Ballas *et. al.* 2015: Delving Deeper into Convolutional Networks for Learning Video Representations](https://arxiv.org/abs/1511.06432)

## Development

This tool is a product of the [Laboratory of Cell Geometry](https://cellgeometry.ucsf.edu/) at the [University of California, San Francisco](https://ucsf.edu).
