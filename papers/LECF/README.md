# LECF

MindSpore implementation for LECF: Recommendation via Learnable Edge Collaborative Filtering

## Introduction

In this work, we proposed a novel variant of CF, which generates recommendation based on the similarity between edges, rather than the simple relationship between objects (users and items) commonly used in previous work. Based on this idea, we designed a new framework LECF, which calculates the weighted sum of edge-edge similarity as the score to rank items.

## Requirements

```bash
mindspore=1.6.1
scipy=1.7.3
numpy=1.21.5
```

## Run

```bash
python main.py --epochs 300 --dataset video10
```

## Reference

```text
@article{xiao2022lecf,
  title={LECF: recommendation via learnable edge collaborative filtering},
  author={Xiao, Shitao and Shao, Yingxia and Li, Yawen and Yin, Hongzhi and Shen, Yanyan and Cui, Bin},
  journal={Science China Information Sciences},
  volume={65},
  number={1},
  pages={1--15},
  year={2022},
  publisher={Springer}
}
```



