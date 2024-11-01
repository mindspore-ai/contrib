# Deep Gradient Compression

We implement the Deep Gradient Compression within the optimizer. We applied the latest feature of the PyTorch 2.0 into the codebase. You can just replace the `SGD` into our `DGC_SGD` to apply the deep gradient compression.

We also introduce the momentum correction into the design so that the optimizer would be avaiable to apply with momentum.

## Features

- Implementation of SGD optimizer with Deep Gradient Compression in PyTorch. 
- Sparsifies and quantizes gradients to reduce communication overhead during distributed training.
- Supports momentum and momentum correction for DGC.

## Configuration Options

The `DGC_SGD` optimizer accepts the following configuration options:

- `lr` (float): Learning rate for SGD. Default: 0.001.
- `momentum` (float): Momentum factor for SGD. Default: 0.
- `dampening` (float): Dampening factor for momentum. Default: 0.
- `weight_decay` (float): Weight decay (L2 penalty) coefficient. Default: 0.
- `nesterov` (bool): Enables Nesterov momentum. Default: False.
- `compress_ratio` (float): Compression ratio for gradient sparsification. Default: 0.01.

Reference

```tex
@article{lin2017deep,
  title={Deep gradient compression: Reducing the communication bandwidth for distributed training},
  author={Lin, Yujun and Han, Song and Mao, Huizi and Wang, Yu and Dally, William J},
  journal={arXiv preprint arXiv:1712.01887},
  year={2017}
}
```

