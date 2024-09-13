## SNUH-mindspore

The [MindSpore](https://www.mindspore.cn/) implementation of "[Integrating Semantics and Neighborhood Information with Graph-Driven Generative Models for Document Retrieval](https://arxiv.org/pdf/2105.13066.pdf)" (ACL 2021).

The PyTorch implementation is available in [this repo](https://github.com/J-zin/SNUH).

### Dependencies

- Ubuntu-x86

- [mindspore 1.2.0 CPU version](https://www.mindspore.cn/install/)

### Datasets

We follow the setting of VDSH [(Chaidaroon and Fang, 2017)](https://arxiv.org/pdf/1708.03436.pdf). Please download the data from [here](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset) and move them into the `./data/` directory.

### Run

The detailed running commands refer to `run.sh`.

### Experimental Results

|         | 16bits | 32bits | 64bits | 128bits |
| :-----: | :----: | :----: | :----: | :-----: |
| Reuters | 80.70  | 83.08  | 84.77  |  85.48  |
|   TMC   | 71.98  | 74.54  | 76.40  |  76.64  |
|  NG20   | 54.75  | 63.75  | 64.83  |  67.55  |

### Cite

```latex
@article{zijing2021snuh,
  author    = {Zijing Ou and
               Qinliang Su and
               Jianxing Yu and
               Bang Liu and
               Jingwen Wang and
               Ruihui Zhao and
               Changyou Chen and
               Yefeng Zheng},
  title     = {Integrating Semantics and Neighborhood Information with Graph-Driven
               Generative Models for Document Retrieval},
  journal   = {arXiv preprint arXiv:2105.13066},
  year      = {2021},
}
```