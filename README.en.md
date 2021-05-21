# mindspore-segmentation

## Introduction

MindSpore Segmentation is an open source semantic segmentation toolbox based on MindSpore. This branch is mainly contribute by GZU.

The master branch works with **MindSPore 1.2**.

![demo image](docs/demo1.png)


## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)]
- [x] [ResNeSt (ArXiv'2020)]
- [x] [MobileNetV2 (CVPR'2018)]
- [x] [MobileNetV3 (ICCV'2019)]

Supported methods:

- [x] [FCN (CVPR'2015/TPAMI'2017)]
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)]
- [x] [PSPNet (CVPR'2017)]
- [x] [DeepLabV3 (CVPR'2017)]
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)]
- [x] [PSANet (ECCV'2018)]
- [x] [DeepLabV3+ (CVPR'2018)]
- [x] [UPerNet (ECCV'2018)]
- [x] [NonLocal Net (CVPR'2018)]
- [x] [EncNet (CVPR'2018)]
- [x] [Semantic FPN (CVPR'2019)]
- [x] [DANet (CVPR'2019)]
- [x] [APCNet (CVPR'2019)]
- [x] [EMANet (ICCV'2019)]
- [x] [CCNet (ICCV'2019)]
- [x] [DMNet (ICCV'2019)]
- [x] [ANN (ICCV'2019)]
- [x] [GCNet (ICCVW'2019/TPAMI'2020)]
- [x] [Fast-SCNN (ArXiv'2019)]
- [x] [OCRNet (ECCV'2020)]
- [x] [DNLNet (ECCV'2020)]
- [x] [PointRend (CVPR'2020)]
- [x] [CGNet (TIP'2020)]


## Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Gitee Issues](https://gitee.com/mind_spore/mindspore-segmentation/issues).

## Contributing

We appreciate all contributions to improve MindSpore Segmentation. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for the contributing guideline.

## Contributors

ZOMI(chenzomi12@gmail.com) / xiaoyisd(524925587@qq.com) / Ronghost(836030680@qq.com) / yehong(2743897969@qq.com) / Dcklin(88522746@qq.com) / Vicrays(384639387@qq.com) / tktkbai(lcl123465@qq.com)/lvbx(a13286625300@163.com)/Han_Junyu(2445939651@qq.com)/ while_bear(2007400050@e.gzhu.edu.cn) /bueng(18407519214@163.com)/Juan(1739347519@qq.com) / IS-2(2691560989@qq.com) / wyh(1063876635@qq.com) / Ricky(veithly@163.com)

## Acknowledgement

MindSpore Segmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{msseg2021,
    title={{MindSporeSegmentation}:Semantic Segmentation Toolbox and Benchmark},
    author={MindSporeSegmentation Contributors},
    howpublished = {\url{https://gitee.com/mind_spore/mindspore-segmentation}},
    year={2021}
}
```
