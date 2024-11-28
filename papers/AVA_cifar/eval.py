import os
import argparse
import random
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
import mindspore.dataset.engine as de

from src.loss import LossNet
from src.network_define import WithLossCell, TrainOneStepCell
from src.config import get_config, save_config, get_logger
from src.datasets import get_train_dataset, get_test_dataset, get_train_test_dataset
from src.cifar_resnet import resnet18, resnet50, resnet101
from src.knn_eval import KnnEval, FeatureCollectCell

random.seed(123)
np.random.seed(123)
de.config.set_seed(123)

parser = argparse.ArgumentParser(description="AVA evaluation")
parser.add_argument("--run_distribute", type=bool, default=False, help="Run distribute, default is false.")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--device_target", type=str, default="Ascend", help="Device target")
parser.add_argument("--train_data_dir", type=str, default="", help="training dataset directory")
parser.add_argument("--test_data_dir", type=str, default="", help="testing dataset directory")
parser.add_argument("--load_ckpt_path", type=str, default="", help="path to load pretrain model checkpoint")
parser.add_argument("--network", type=str, default="resnet18", help="network architecture: (resnet18, resnet50, resnet101)")
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    temp_path = ''

    device_id = args_opt.device_id
    device_num = args_opt.device_num
    print("device num:{}".format(device_num))
    print("device id:{}".format(device_id))

    # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=device_id)

    train_data_dir = os.path.join(temp_path, args_opt.train_data_dir)
    test_data_dir = os.path.join(temp_path, args_opt.test_data_dir)

    print("start create dataset...")

    # eval_dataset contains train dataset and test dataset, which is used for knn eval
    eval_dataset = get_train_test_dataset(train_data_dir=train_data_dir, test_data_dir=test_data_dir,
                                          batchsize=100, epoch=1)

    eval_dataset_batch_num = int(eval_dataset.get_dataset_size())


    if args_opt.network == 'resnet18':
        resnet = resnet18(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    elif args_opt.network == 'resnet50':
        resnet = resnet50(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    elif args_opt.network == 'resnet101':
        resnet = resnet101(low_dims=config.low_dims, training_mode=True, use_MLP=config.use_MLP)
    else:
        raise ("net work config error!!!")

    load_checkpoint(args_opt.load_ckpt_path, net=resnet)
    print("load ckpt from {}".format(args_opt.load_ckpt_path))

    loss = LossNet(temp=config.sigma)

    eval_network = FeatureCollectCell(resnet)

    net_with_loss = WithLossCell(resnet, loss)

    net = TrainOneStepCell(net_with_loss, optimizer=None)


    model = Model(net, metrics={'knn_acc': KnnEval(batch_size=config.batch_size, device_num=1)},
                  eval_network=eval_network)

    output = model.eval(eval_dataset)
    knn_acc = float(output["knn_acc"])
    print("The knn result is {}.".format(knn_acc))