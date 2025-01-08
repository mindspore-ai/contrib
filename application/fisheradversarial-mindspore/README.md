# Detect adversarial examples using the Fisher information

This implements a method for the detection of adversarial examples based on the Fisher information, presented in the Neurocomputing article *Inspecting adversarial examples using the fisher information* [1]. There is also a (slightly less elaborated) arXiv preprint available [2]. 

## Installation
You need to have [mindspore](https://www.mindspore.cn/install/) installed in your virtual environment, in order to run this software.


## Usage

**Note**: The method is currently designed for inputs `x` of batch size 1. That is, it is assumed that `x.shape[0] == 1`.

The main file of this package is the module `fisherform`:
```
import mindspore 
import mindspore.nn as ms_nn
import mindspore.ops as ops
import fisherform
import fishertrace
```
There is also a module `fishertrace` contained in the package that implements adversarial detection based on the trace of the Fisher information matrix, as explained in [1]. To check that the software works you might want to pick some random objects:
```
ms_net = ms_nn.SequentialCell(ms_nn.Dense(10,5),ms_nn.Tanh(),ms_nn.Dense(5,3),ms_nn.LogSoftmax(axis=1))
ms_x = ops.randn((1,10))

```
For a (trained) pytorch model `ms_net` with log-softmax output and an input `ms_x` with batch size 1, first compute a corresponding direction `v`. For instance, as explained in [1], you can take the gradient of the maximal output node, that is
```
def forward(ms_x):
    return ms_net(ms_x).max()
grad_fn = mindspore.value_and_grad(forward,has_aux=False)
ms_out = ms_net(ms_x).max()
output, inputs_gradient = grad_fn(ms_x)
ms_v = []
for ms_par in ms_net.get_parameters():
    ms_v.append(ms_par.value().copy())
```
Computing the fisher form from [1] for adversarial detection can then be achieved via
```
fisherform.numeric_fisher_form(ms_x, ms_net, ms_v)
```
This 0 dimensional tensor measures how *unusual* the input is, compared to the learned parameters of `ms_net`. If you are using a trained network, and not a random one as above, this number will go up once you feed adversarial inputs `ms_x`. 

The second quantity introduced in [1] is the Fisher information sensitivity (FIS). The FIS measures, loosely speaking, the contribution of the input nodes to the unusualness and will thus have the same dimensionality as the input `ms_x`. To compute it type
```
fisherform.fisher_information_sensitivity(ms_x, ms_net, ms_v) 
```
If you used the random net from above, then these numbers will probably be all of a similar magnitude. If you are using a trained network, and images as input, than you might plot the FIS - compare [1] for images.



## Citation

```@article{martin2020inspecting,
  title={Inspecting adversarial examples using the Fisher information},
  author={Martin, J{\"o}rg and Elster, Clemens},
  journal={Neurocomputing},
  volume={382},
  pages={80--86},
  year={2020},
  publisher={Elsevier}
}
```


## References
[1] Martin, J., & Elster, C. (2020). Inspecting adversarial examples using the Fisher information. Neurocomputing, 382, 80-86. https://doi.org/10.1016/j.neucom.2019.11.052

[2] https://arxiv.org/abs/1909.05527

## License

 copyright: Jörg Martin(PTB), 2020.
 
 This software is licensed under the BSD-like license:

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.

 ## DISCLAIMER
 
 This software was developed at Physikalisch-Technische Bundesanstalt
 (PTB). The software is made available "as is" free of cost. PTB assumes
 no responsibility whatsoever for its use by other parties, and makes no
 guarantees, expressed or implied, about its quality, reliability, safety,
 suitability or any other characteristic. In no event will PTB be liable
 for any direct, indirect or consequential damage arising in connection
