# Contents

[toc]

# [Diffnet++ Description](#contents)

Diffnet++ model is a classical model in Social Recommendation area.  This is an implementation of Diffnet++ as described in the [DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation](https://arxiv.org/pdf/2002.00844.pdf) paper.

# [Model Architecture](#contents)

Diffnet++ model fuses both influence diffusion in the social network $G_S$ and interest diffusion in the interest network $G_I$ for social recommendation. The architecture of DiffNet++ contains four main parts: an embedding layer, a fusion layer, the influence and interest diffusion layers, and a rating prediction layer.

# [Dataset](#contents)

The Yelp dataset are used for model training and evaluation.

## Yelp

Yelp is a well-known on-line location based social network, where users could make friends with others and review restaurants.

Yelp dataset contains 204,448 ratings of 38,342 items by 17,237 users. We transform the original ratings to binary values. If the rating value is larger than 3, we transform it into 1, otherwise it equals 0. Each line of this file  has the following format:

```bash
user_id item_id label
```

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
python train.py
python eval.py
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```bash
└── Diffnet_plus_plus
  ├── check_point
    ├── conf
        ├── yelp_Diffnet_plus_plus_ms.ini
    ├── data
    ├── datahelper.py
    ├── DataModule.py
    ├── dataset.py
    ├── eval.py
    ├── train.py
    ├── Metrics.py
    ├── ParserConf.py
    ├── README.md
```

## [Training Process](#contents)

```bash
python train.py
```

After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

```bash
Epoch:1
Loss:66502.18
Epoch:2
Loss:66335.62
...
```

The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "./check_point/Diffnetplus_220.ckpt".

```bash
python eval.py
```

The above python command will run in the background. You can view the results on the screen. The accuracy of epoch 220 below is the best:

```bash
HR: 0.314 NDCG: 0.277
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters        | GPU                   |
| ----------------- | --------------------- |
| Model Version     | Diffnet++             |
| Resource          | GPU                   |
| Uploaded Date     | 1/9/2021              |
| MindSpore Version | 1.2.1                 |
| Dataset           | Yelp                  |
| batch_size        | Full batch            |
| Accuracy          | HR: 0.314 NDCG: 0.277 |

# [Description of Random Situation](#contents)

There are three random situations:

- Generating negative samples randomly.

- Initialization of some model weights.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
