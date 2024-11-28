# Integrated-Gradient-Mindspore

This code is a minimum implementation of Integrated-Gradient which is avaliable at https://github.com/shyhyawJou/Integrated-Gradient-Pytorch ,introduced in the paper https://arxiv.org/abs/1703.01365 by  Mukund Sundararajan,Ankur Taly and Qiqi Yan.

## Requirements

- Python 3.9
- MindSpore 2.3
- mindcv 0.3.0

### Running on Huawei Cloud ModelArts

This implementation can be run on Huawei Cloud’s ModelArts platform using the *Guizhou 1* node. For this, select the environment image:

```
mindspore_2.3.0-cann_8.0.rc1-py_3.9-euler_2.10.7-aarch64-snt9b
```

with the following commind:

````
pip install mindcv
````

## Usage

To get the similar picture in the original repo, run the following file:

```
python show.py -img assets/n01669191_46.JPEG -step 20
```

