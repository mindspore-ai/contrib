# Bi-Tempered-Loss (MindSpore)

This code is a mindspore implementation of Bi-Tempered-Loss which is avaliable at https://github.com/mlpanda/bi-tempered-loss-pytorch ,introduced in the paper https://paperswithcode.com/paper/robust-bi-tempered-logistic-loss-based-on by Ehsan Amid. 

## Requirements

- Python 3.9
- MindSpore 2.3

### Running on Huawei Cloud ModelArts

This implementation can be directly run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

```
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
```

## Usage

To test the loss function with similar values to the test in the original repo, run the following file:

```
python test_loss.py
```

