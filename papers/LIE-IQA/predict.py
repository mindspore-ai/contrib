#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""predict"""

import os
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from networks.model import RetinexLieIqaNetVgg
from utils.datasets import LieIqaDataset
from utils.tools import create_iqa_testset


ds.config.set_seed(58)
root_dir = os.path.dirname(os.path.realpath(__file__))
lledataset = LieIqaDataset(root_dir=os.path.join(root_dir, 'images'), txt_file='test_file.txt')
lledataset = ds.GeneratorDataset(lledataset, ['x', 'y', 'x_name'], shuffle=False)
lledataset = create_iqa_testset(lledataset, batch_size=1)
dataset_iter = lledataset.create_dict_iterator(output_numpy=True)

model = RetinexLieIqaNetVgg(load_weight=True)
net_dict = load_checkpoint(os.path.join(root_dir, 'checkpoint/Retinex_LIEIQANet.ckpt'))
# net_dict = load_checkpoint(os.path.join(root_dir, 'checkpoint/Retinex_ComIQANet.ckpt'))
param_not_load = load_param_into_net(model, net_dict, strict_load=True)
print('params not load to net: ', param_not_load)

for _, data in enumerate(dataset_iter):
    lle_img = Tensor(data['x'])
    ref_img = Tensor(data['y'])
    lle_name = str(data['x_name'][0], encoding='utf-8')
    score = model(lle_img, ref_img)
    score = score.asnumpy()
    print('{:32}: {:.3f} '.format(lle_name, score))
