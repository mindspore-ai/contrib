# An Unofficial implementation of 'ACGNet: Action Complement Graph Network for Weakly-supervised Temporal Action Localization'

> This is the MindSpore implementation of the [original repository](https://github.com/xjtupanda/ACGNet)

```
File Structure:
    - model.py contains model building code for ACGNet.
    - loss.py contains code implementation for 'Easy Positive Mining' loss in the paper.
    - main.py contains a test script to check the correctness of the code
```
Note that the memory footprint could be huge, so batch_size might need to be set a lot smaller than original framework, or cut off num_segments by a large margin.

## Code Example
```Python
from model import ACGNet
from loss import loss_EPM

for data, label in data_loader:
    orig_F, new_F, A_prime = acg_net(_data)
    _, score_act = net(new_F)
    
    loss_epm = loss_EPM(A_prime, orig_F, new_F, score_act)
```