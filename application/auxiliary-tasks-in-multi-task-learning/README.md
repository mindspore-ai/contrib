# AutomaticWeightedLoss

A MindSpore implementation of Liebel L, Körner M. [Auxiliary tasks in multi-task learning](https://arxiv.org/pdf/1805.06334)[J]. arXiv preprint arXiv:1805.06334, 2018.

The above paper improves the paper "[Multi-task learning using uncertainty to weigh losses for scene geometry and semantics](http://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)" to avoid the loss becoming negative during training.

## Requirements

* Python
* MindSpore

## How to Train with Your Model

* Clone the repository

```bash
git clone git@github.com/YourUsername/AutomaticWeightedLoss_MindSpore.git
```

* Create an `AutomaticWeightedLoss` module

```python
from AutomaticWeightedLoss import AutomaticWeightedLoss

awl = AutomaticWeightedLoss(2)  # we have 2 losses
loss1 = Tensor(1.0, mindspore.float32)
loss2 = Tensor(2.0, mindspore.float32)
loss_sum = awl(loss1, loss2)
```

* Create an optimizer to learn weight coefficients

```python
import mindspore.nn as nn

model = Model() # your model
optimizer = nn.Adam(params=[
                {'params': model.trainable_params()},
                {'params': awl.trainable_params(), 'weight_decay': 0}
            ])
```

* A complete example

```python
import mindspore.nn as nn
from AutomaticWeightedLoss import AutomaticWeightedLoss

model = Model() # your model

awl = AutomaticWeightedLoss(2)  # we have 2 losses
loss_fn1 = nn.MSELoss()  # Example loss function 1
loss_fn2 = nn.CrossEntropyLoss()  # Example loss function 2

# Create optimizer with learnable parameters
optimizer = nn.Adam(params=[
                {'params': model.trainable_params()},
                {'params': awl.trainable_params(), 'weight_decay': 0}
            ])

for epoch in range(epochs):
    for data, label1, label2 in data_loader:
        # forward
        pred1, pred2 = model(data)
        # calculate losses
        loss1 = loss_fn1(pred1, label1)
        loss2 = loss_fn2(pred2, label2)
        # weigh losses
        loss_sum = awl(loss1, loss2)
        # backward
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
```

## Reference

The implementation is adapted from a PyTorch implementation of AutomaticWeightedLoss available here: 
[https://github.com/Mikoto10032/AutomaticWeightedLoss](https://github.com/Mikoto10032/AutomaticWeightedLoss)

