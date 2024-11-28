# CSD: Contrastive Self-Distillation

Mindspore implementation for ***Towards Compact Single Image Super-Resolution via Contrastive Self-distillation*** in IJCAI 2021. Please read our [paper](https://arxiv.org/abs/2105.11683) for more details.

For original PyTorch implementation please refer to [github](https://github.com/Booooooooooo/CSD).

## Abstract

Convolutional neural networks (CNNs) are highly successful for super-resolution (SR) but often require sophisticated architectures with heavy memory cost and computational overhead, significantly restricts their practical deployments on resource-limited devices. In this paper, we proposed a novel contrastive self-distillation (CSD) framework to simultaneously compress and accelerate various off-the-shelf SR models. In particular, a channel-splitting super-resolution network can first be constructed from a target teacher network as a compact student network. Then, we propose a novel contrastive loss to improve the quality of SR images and PSNR/SSIM via explicit knowledge transfer. Extensive experiments demonstrate that the proposed CSD scheme effectively compresses and accelerates several standard SR models such as EDSR, RCAN and CARN.

![model](https://gitee.com/wyboo/csd_-mind-spore/raw/main/images/model.png)

## Results

![tradeoff](https://gitee.com/wyboo/csd_-mind-spore/raw/main/images/tradeoff.png)

![table](https://gitee.com/wyboo/csd_-mind-spore/raw/main/images/table.png)

![visual](https://gitee.com/wyboo/csd_-mind-spore/raw/main/images/visual.png)

## Dependencies

- Python == 3.7.5

- MindSpore: https://www.mindspore.cn/install

- matplotlib

- imageio

- tensorboardX

- opencv-python

- scipy

- scikit-image

## Train

### Prepare data

We use DIV2K training set as our training data.

About how to download data, you could refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Train baseline model

```bash
# train teacher model
python -u train.py --dir_data LOCATION_OF_DATA --data_test Set5 --test_every 1 --filename edsr_baseline --lr 0.0001 --epochs 5000
```

```bash
# train student model (width=0.25)
python -u train.py --dir_data LOCATION_OF_DATA --data_test Set5 --test_every 1 --filename edsr_baseline025 --lr 0.0001 --epochs 5000 --n_feats 64
```

### Train CSD

VGG pre-trained on ImageNet is used in our contrastive loss. Due to copyright reasons, the pre-trained VGG cannot be shared publicly.

```bash
python -u csd_train.py --dir_data LOCATION_OF_DATA --data_test Set5 --test_every 1 --filename edsr_csd --lr 0.0001 --epochs 5000 --ckpt_path ckpt/TEACHER_MODEL_NAME.ckpt --contra_lambda 200
```

- `--neg_num`: specify the number of negative samples. Here we set default neg_num=10.

## Test

```bash
python eval.py --dir_data LOCATION_OF_DATA --test_only --ext "img" --data_test B100 --ckpt_path ckpt/MODEL_NAME.ckpt --task_id 0 --scale 4
```

- `--data_test`: specify the test set. Here we test on B100 test set.

## Citation

If you find the code helpful in you research or work, please cite as:

```@inproceedings{wu2021contrastive,
@misc{wang2021compact,
      title={Towards Compact Single Image Super-Resolution via Contrastive Self-distillation},
      author={Yanbo Wang and Shaohui Lin and Yanyun Qu and Haiyan Wu and Zhizhong Zhang and Yuan Xie and Angela Yao},
      year={2021},
      eprint={2105.11683},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

For the training part of the MindSpore version we referred to [DBPN-MindSpore](https://gitee.com/amythist/DBPN-MindSpore/tree/master), [ModelZoo-RCAN](https://gitee.com/mindspore/models/tree/master/research/cv/RCAN) and the official [tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html). We thank the authors for sharing their codes.