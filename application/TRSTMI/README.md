# TRST-MI

This package contains code for minimizing the coherence of complex matrices.

The file `trstmi.py` contains the main script used for generating low coherence matrices.

## Function Arguments

| Argument | Description |
| ----------- | ----------- |
| `dim1` | lower bound on dimension |
| `dim2` | upper bound on dimension |
| `num1` | lower bound on number of points |
| `num2` | upper bound on number of points |
| `trials` | number of different starting random initializations |
| `tol` | stopping gradient tolerance |
| `proc` | takes the value `cpu` or `gpu` |
| `verbose` | takes the value `0` or `1` |

## Software Requirements

This code runs on Python 3. You will need Mindspore installed. If you would like to use GPU parallelization you will need a version of Mindspore installed supporting your hardware. The trust region optimizer used in the program is a simple implementation using MindSpore, and can be found [here](https://github.com/vchoutas/torch-trust-ncg). 

## Questions

If you have any questions about the code, please contact Carlos Saltijeral (carlossaltrev@gmail.com) or


Josiah Park\
Texas A&M University\
Department of Mathematics\
j.park@tamu.edu
