# Importance weighted autoencoder (MindSpore)

This code is a minimum implementation of importance weighted autoencoder which is avaliable at https://github.com/andrecavalcante/iwae ,introduced in the paper https://arxiv.org/abs/1509.00519 by Yuri Burda, Roger Grosse, Ruslan Salakhutdinov.

## Requirements

- Python 3.9
- MindSpore 2.3
- Mindvision
- Numpy

### Running on Huawei Cloud ModelArts

This implementation can be run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

```
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
```

Add the following command to the terminal:

```
pip install mindvision
```

## Datasets

The MNIST dataset can be obtained from https://yann.lecun.com/exdb/mnist/, and should be extracted into the corresponding dataset folder.The files inside the folder should be organized as follows:

```
--data
	--train-images-idx3-ubyte
	--train-labels-idx1-ubyte
	--t10k-images-idx3-ubyte
	--t10k-labels-idx1-ubyte
```

## Usage

To test the loss with similar values to the test in the original repo, run the following file:

```
python iwae.py
```

