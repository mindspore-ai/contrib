# MindSeg

A mirror repository of [MXNetSeg](https://github.com/BebDong/MXNetSeg) based on the HUAWEI [MindSpore](https://www.mindspore.cn/en) framework.

## Environment

In this project, we adopt:

- Python 3.7.5
- CUDA 10.1
- cuDNN 7.6.5
- Mindspore-gpu 1.0.1

Refer to [MindSpore Installation Guide](https://www.mindspore.cn/install/en) for details.

## Usage

1. Convert the dataset to .mindrecord files

```shell
python build_seg_data.py --data-name Cityscapes --dst-path tmp_data/train.mindrecord --num-shard 4 --shuffle
```

2. Run the `train.py` script

```shell
# for multiple GPUs
mpirun -n 4 python train.py --train-dir experiment --data-file tmp_data/train.mindrecord0 --batch-size 8 --crop-size 1024 --ignore-label 19 --num-classes 19 --epochs 240 --lr-type poly --base-lr 0.04 --wd 4.e-4 --model eprnet --device-target GPU --distributed
```

3. Run the `eval.py` for performance evaluation

```shell
python eval.py --data-name Cityscapes --batch-size 16 --crop-size 1024 --scales 0.5 0.75 1.0 1.25 1.75 2.0 --flip --ignore-label 19 --num-classes 19 --eval-split val --model eprnet --checkpoint experiment/ckpt/eprnet-10_27.ckpt
```

### Remarks

- The `ResizeBilinear` and `PReLU` operators of MindSpore do not support NVIDIA GPU for now
- When evaluating model performance, the `ignore-label` indicator is temporarily not supported, which will be improved in a future version.