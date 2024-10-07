## Riemann Noise Injection - PyTorch

A module for modeling GAN noise injection based on Riemann geometry, as described in Ruili Feng, Deli Zhao, and Zheng-Jun Zha's paper <a href="http://proceedings.mlr.press/v139/feng21g/feng21g.pdf">"Understanding Noise Injection in GANs"</a>.

```python
import torch
from riemann_noise_pytorch import RiemannNoise

class Generator(torch.nn.Module):
    def __init__(self):
        ...
        self.riemann_noise = RiemannNoise(128, torch.device("cuda"))
        ...
    def forward(self, x):
        out = self.DownBlock(x)
        out = self.resblock(out)
        out = self.riemann_noise(out)
        out = self.UpBlock(out)
        return out
```

## Citations

```
@InProceedings{pmlr-v139-feng21g,
  title = 	 {Understanding Noise Injection in GANs},
  author =       {Feng, Ruili and Zhao, Deli and Zha, Zheng-Jun},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3284--3293},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/feng21g/feng21g.pdf},
  url = 	 {https://proceedings.mlr.press/v139/feng21g.html},
  abstract = 	 {Noise injection is an effective way of circumventing overfitting and enhancing generalization in machine learning, the rationale of which has been validated in deep learning as well. Recently, noise injection exhibits surprising effectiveness when generating high-fidelity images in Generative Adversarial Networks (GANs) (e.g. StyleGAN). Despite its successful applications in GANs, the mechanism of its validity is still unclear. In this paper, we propose a geometric framework to theoretically analyze the role of noise injection in GANs. First, we point out the existence of the adversarial dimension trap inherent in GANs, which leads to the difficulty of learning a proper generator. Second, we successfully model the noise injection framework with exponential maps based on Riemannian geometry. Guided by our theories, we propose a general geometric realization for noise injection. Under our novel framework, the simple noise injection used in StyleGAN reduces to the Euclidean case. The goal of our work is to make theoretical steps towards understanding the underlying mechanism of state-of-the-art GAN algorithms. Experiments on image generation and GAN inversion validate our theory in practice.}
}
```

```
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```