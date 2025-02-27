# Unofficial PyTorch implementation of "Deep Weakly-supervised Anomaly Detection" paper

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


## Usage

```python
from pro import PRO
from demo import Feature_Encoder, Classifer

# normal - tensor of normal samples with shape [n_pos_samples, n_features]
# anomalies - tensor of anomalies with shape [n_neg_samples, n_features]

model = PRO(normal, anomalies, Feature_Encoder(n_features), Classifier())
model.fit(epoches=n_epoches)

preds = model.predict(test_x) # 0 - normal sample, 1 - anomaly

```


## Authors

```
@article{pro,
  title={Deep Weakly-supervised Anomaly Detection},
  author={Pang, Guansong and Shen, Chunhua and Jin, Huidong and Anton, van den Hengel},
  journal={arXiv preprint arXiv:1910.13601},
  year={2020}
}
```

## Contacts

Artem Ryzhikov, LAMBDA laboratory, Higher School of Economics, Yandex School of Data Analysis

**E-mail:** artemryzhikoff@yandex.ru

**Linkedin:** https://www.linkedin.com/in/artem-ryzhikov-2b6308103/

**Link:** https://www.hse.ru/org/persons/190912317
