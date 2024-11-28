# Local quantum annealing for Ising optimisation problems

 This code is a mindspore implementation of energy minimisation of Ising Hamiltonians which is avaliable at https://github.com/josephbowles/localquantumannealing. It performs energy minimisation of Ising Hamiltonians as described in https://physics.paperswithcode.com/paper/quadratic-unconstrained-binary-optimisation.

## Requirements

- Python 3.9
- MindSpore 2.3
- tqdm

### Running on Huawei Cloud ModelArts

This implementation can be directly run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

````
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
````

## Usage

Test the LQA model with the command: 

````
python test.py --testmode 1   
````

The parameter `testmode` sets the model type. (1 for LQA, 0 for LQA_basic)

