import torch
from mindspore import Tensor, save_checkpoint


par_dict = torch.load('best.pth', map_location='cpu')
params_list = []
for name in par_dict:
    param_dict = {}
    parameter = par_dict[name]
    param_dict['name'] = name
    param_dict['data'] = Tensor(parameter.numpy())
    params_list.append(param_dict)
save_checkpoint(params_list,  'best_ms.ckpt')
