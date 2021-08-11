# LIE-IQA

The [MindSpore](https://www.mindspore.cn/) implementation of LIE-IQA.

You can get [Pytorch](https://pytorch.org/) the implementation here [LIE-IQA-pytorch](https://github.com/yiumac/LIE-IQA). It is worth noting that the MindSpore implementation of Image Intrinsic Decomposition (IID) Module is different from the Pytorch implementation , but there is not much difference in performance. Please refer to the specific code for details.

## Requirements

+ Python 3.7.5
+ MindSpore 1.2.1
+ CUDA 10.1

## Quality Assessment for Enhanced Low-light Image

+ LIE-IQA Framework

  <img src="fig/LIE-IQA-Framework.png" width="100%" />

+ Performance on LIE-IQA Dataset

  <img src="fig/performance-LIE-IQA-Dataset.png" width="80%"/>

+ Performance on General Scene IQA Dataset (LIVE, MDID, CSIQ)

  <img src="fig/performance-IQA-Dataset.png" width="100%" />

## Quality Optimization for Low-light Image Enhancement

+ Optimization framework

  <img src="fig/optimization-framework.png" width="80%" />

## Predict

+ download the checkpoint from [Baidu Drive](https://pan.baidu.com/s/1oqE7VQHpmtfyjMuTQqeTfw) (ae9a)

```bash
# download the checkpoint
# run LIEIQA_predict.py
python LIEIQA_predict.py
```
