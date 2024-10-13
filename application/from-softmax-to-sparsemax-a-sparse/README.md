# Sparsemax Activation Function in MindSpore

This repository contains an implementation of the Sparsemax activation function in MindSpore, inspired by the paper:  
**[From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068)**  
by André F. T. Martins and Ramón Fernandez Astudillo.

## Overview

Sparsemax is a variation of the Softmax function that results in sparse probabilities, meaning it can output zero probabilities for certain classes, making it particularly useful in models where sparsity is desired, such as attention mechanisms and multi-label classification tasks.

## Tested Environment

- **MindSpore version**: 2.3.1


## Reference

The implementation is adapted from a PyTorch implementation of Sparsemax available here:  
[https://github.com/kriskorrel/sparsemax-pytorch](https://github.com/kriskorrel/sparsemax-pytorch)

## Example Usage

Below is a simple example demonstrating how to use the Sparsemax activation function in MindSpore.

```python
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np  
from sparsemax import Sparsemax

# Initialize the Sparsemax and Softmax functions
sparsemax = Sparsemax(dim=1)
softmax = nn.Softmax(axis=1)

# Generate some random logits
logits_np = np.random.randn(2, 5).astype(np.float32)

# Convert the numpy array to a MindSpore Tensor
logits = Tensor(logits_np, mindspore.float32)

print("\nLogits")
print(logits)

# Compute Softmax probabilities
softmax_probs = softmax(logits)
print("\nSoftmax probabilities")
print(softmax_probs)

# Compute Sparsemax probabilities
sparsemax_probs = sparsemax(logits)
print("\nSparsemax probabilities")
print(sparsemax_probs)
```
