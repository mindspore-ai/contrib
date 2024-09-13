# Ucolor

> This is the implementation of Ucolor using [MindSpore](https://www.mindspore.cn/)

## Requirements

1. python 3.7.5
2. minspore-gpu 1.3.0
3. cuda 10.1

You'd better install the mindspore-gpu follow the [official instruction](https://www.mindspore.cn/install).

You can create a conda environment to run our code like this:

```bash
conda env create -f ./mindspore1.3.yaml
```

## Train

1. Deownload the [pretrain vgg](https://www.mindspore.cn/resources/hub/details?MindSpore/ascend/v1.2/vgg16_v1.2_imagenet2012) for mindspore.
2. Put your data in the data folder.
3. You are ready to train your own UColor Model!

```bash
python ./train.py
```

Use `python ./train.py --help` for more details.

## Test

Put your data in the data folder.

```bash
python ./test.py
```

Use `python ./test.py --help` for more details.

## Acknowledgement

Thanks to the minspore team who maintain the latest version of the [framework](https://www.mindspore.cn/),
and also the [model zoo](https://gitee.com/mindspore/models).
