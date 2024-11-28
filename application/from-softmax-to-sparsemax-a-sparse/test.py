# Example usage

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np  
from sparsemax import Sparsemax

# Initialize the Sparsemax and Softmax functions
sparsemax = Sparsemax(dim=1)
softmax = nn.Softmax(axis=1)


logits_np = np.random.randn(2, 5).astype(np.float32)


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
