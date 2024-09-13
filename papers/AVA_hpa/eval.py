import argparse
import random
import numpy as np

from mindspore import context
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import mindspore.dataset.engine as de
from src.datasets import makeup_dataset
from src.resnet import resnet18, resnet50, resnet101
from src.network_define_eval import EvalCell, EvalMetric
from src.loss import BCELoss

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="eval")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument('--device_target', type=str, default="Ascend", help='Device target')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--ckpt_path', type=str, default="", help='model checkpoint path')
parser.add_argument("--model_arch", type=str, default="resnet18", help='model architecture')
parser.add_argument("--data_dir", type=str, default="",
                    help='dataset path')
parser.add_argument("--classes", type=int, default=10, help='class number')
parser.add_argument("--save_eval_path", type=str, default=".", help='path to save eval result')
args_opt = parser.parse_args()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)
    ckpt_path = args_opt.ckpt_path
    data_dir = args_opt.data_dir

    if args_opt.model_arch == 'resnet18':
        resnet = resnet18(pretrain=False, classes=args_opt.classes)
    elif args_opt.model_arch == 'resnet50':
        resnet = resnet50(pretrain=False, classes=args_opt.classes)
    elif args_opt.model_arch == 'resnet101':
        resnet = resnet101(pretrain=False, classes=args_opt.classes)
    else:
        raise ("Unsupported net work!")

    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(resnet, param_dict)
    test_dataset = makeup_dataset(data_dir=data_dir, mode='test', batch_size=1, bag_size=20, classes=args_opt.classes,
                                  num_parallel_workers=8)
    test_dataset.__loop_size__ = 1

    test_dataset_batch_num = int(test_dataset.get_dataset_size())

    loss = BCELoss(reduction='mean')
    test_network = EvalCell(resnet, loss)
    model = Model(test_network, metrics={'results_return': EvalMetric(path=args_opt.save_eval_path)},
                  eval_network=test_network)
    result = model.eval(test_dataset)
    print(result)
