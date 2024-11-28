This is a mindspore implementation of "Simba: A Scalable Bilevel Preconditioned Gradient Method for Fast Evasion of Flat Areas and Saddle Points"
You can check the paper here: <https://math.paperswithcode.com/paper/simba-a-scalable-bilevel-preconditioned>

Simba是一种加速函数逃离鞍部的优化器。特别适用于处理具有平坦区域和鞍点的目标函数。Simba 通过引入双层预条件梯度方法，旨在加速优化过程，特别是在目标函数具有平坦区域和鞍点的情况下。

Simba引入了一个粗糙梯度的预调节器，并将其SVD分解后与原梯度结合，作为新的优化方向，通过引入双层预条件梯度方法，能够更好地处理这些区域，从而加速优化过程。

由于Simba的运算量高于Adam，Adagrad等常规优化方法，更适合作为函数鞍部的局部优化器，不适合作为全局优化器。
