# DoRA(MindSpore)

This code is a minimum implementation of weight-decomposed low-rank adaptation which is avaliable at https://github.com/catid/dora ,introduced in the paper https://arxiv.org/pdf/2402.09353 by Shih-Yang Liu.

## Requirements

- Python 3.9
- MindSpore 2.3

### Running on Huawei Cloud ModelArts

This implementation can be run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

```
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
```

## Usage

To test the loss with similar values to the test in the original repo, run the following file:

```
python dora.py
```

