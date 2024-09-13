# RHFL

This is the MindSpore implementation of RHFL in the following paper.

CVPR 2022: Robust Federated Learning with Noisy and Heterogeneous Client

# [RHFL Description](#contents)

RHFL (Robust Heterogeneous Federated Learning) is a federated learning framework to solve the robust federated learning problem with noisy and heterogeneous clients:

1) Aligning the logits output distributions in heterogeneous federated learning.

2) Local noise learning with a noise-tolerant loss function.

3) Client confidence re-weighting for external noise.

# [Framework Architecture](#contents)

![image-20220629214205667](https://cdn.jsdelivr.net/gh/xiye7lai/cdn/bg/pic/image-20220629214205667.png)

# [Dataset](#contents)

Our experiments are conducted on two datasets, Cifar10 and Cifar100. We set public dataset on the server as a subset of Cifar100, and randomly divide Cifar10 to different clients as private datasets.

Dataset used: [CIFAR-10、CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

CIFAR10

- 60,000 32*32 colorful images in 10 classes

CIFAR100

- 60,000 32*32 colorful images in 100 classes(20 superclasses)

Note: Data will be processed in init_data.py

# [Environment Requirements](#contents)

Hardware(GPU）

- Prepare hardware environment GPU processor.

Framework

- [MindSpore](https://gitee.com/mindspore/mindspore)

For more information, please check the resources below：

- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# init public data and local data
python Dataset/init_data.py
# pretrain local models
python Network/pretrain.py
# RHFL
python RHFL/RHFL.py
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
├── RHFL
    ├── Dataset
        ├── datasets
        ├── cifar10.py
        ├── init_data.py
        ├── utils.py
    ├── Network
        ├── Models_Def
            ├── efficientnet.py
            ├── resnet.py
            ├── shufflenet.py
        ├── pretrain.py
    ├── RHFL
        ├── RHFL.py
    ├── loss.py
    ├── README.md
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

In the heterogeneous model scenario, we assign four different networks:ResNet10,ResNet12,ShuffleNet,EfficientNet

（Our experimental setting is not exactly the same as the original, because the mobilenetv2 in MindSpore model zoo does not fit the loss function, and another model efficientnet was chosen to replace it after testing.）

#### noise rate = 0.1 noise type = pairflip

|                  | θ1    | θ2    | θ3    | θ4    | Avg   |
| ---------------- | ----- | ----- | ----- | ----- | ----- |
| CE               | 67.17 | 66.45 | 52.05 | 49.39 | 58.76 |
| SL               | 64.41 | 62.97 | 54.09 | 51.69 | 58.29 |
| RHFL(SL+HFL+CCR) | 67.73 | 69.46 | 60.07 | 59.64 | 64.23 |

#### noise rate = 0.1 noise type = symmetric

|                  | θ1    | θ2    | θ3    | θ4    | Avg   |
| ---------------- | ----- | ----- | ----- | ----- | ----- |
| CE               | 66.20 | 67.73 | 52.43 | 50.19 | 59.14 |
| SL               | 64.40 | 64.68 | 53.48 | 50.55 | 58.28 |
| RHFL(SL+HFL+CCR) | 69.61 | 70.49 | 59.74 | 59.51 | 64.84 |

#### noise rate = 0.2 noise type = pairflip

|                  | θ1    | θ2    | θ3    | θ4    | Avg   |
| ---------------- | ----- | ----- | ----- | ----- | ----- |
| CE               | 62.35 | 62.85 | 47.08 | 47.05 | 54.83 |
| SL               | 62.20 | 62.55 | 52.08 | 48.60 | 56.36 |
| RHFL(SL+HFL+CCR) | 65.83 | 66.79 | 58.64 | 58.36 | 62.40 |

#### noise rate = 0.2 noise type = symmetric

|                  | θ1    | θ2    | θ3    | θ4    | Avg   |
| ---------------- | ----- | ----- | ----- | ----- | ----- |
| CE               | 60.76 | 62.87 | 49.54 | 48.93 | 55.52 |
| SL               | 64.73 | 63.71 | 53.75 | 51.01 | 58.30 |
| RHFL(SL+HFL+CCR) | 67.72 | 68.23 | 57.10 | 56.78 | 62.46 |

# [Citation](#contents)

```citation
@inproceedings{fang2022robust,
  title={Robust Federated Learning With Noisy and Heterogeneous Clients},
  author={Fang, Xiuwen and Ye, Mang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10072--10081},
  year={2022}
}
```
