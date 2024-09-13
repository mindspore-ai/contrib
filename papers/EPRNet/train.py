# coding=utf-8
# adapted from:
# https://github.com/mindspore-ai/mindspore/blob/master/model_zoo/official/cv/deeplabv3/train.py

import os
import sys
import shutil
import argparse
from mindspore import nn, context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig

# &PATH
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)

from mindseg.data import TransformSegDataset
from mindseg.models import get_model_by_name
from mindseg.nn import SoftmaxCrossEntropyLoss
from mindseg.tools import lr_scheduler, makedir_p

set_seed(1)


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


def parse_args():
    parser = argparse.ArgumentParser('mindspore semantic segmentation training')

    # experiments dir
    parser.add_argument('--train-dir', type=str, default='experiment',
                        help='relative path to root dir')

    # dataset
    parser.add_argument('--data-file', type=str, default='tmp_data/train.mindrecord0',
                        help='relative path to root dir')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--min-scale', type=float, default=0.5)
    parser.add_argument('--max-scale', type=float, default=2.0)
    parser.add_argument('--ignore-label', type=int, default=19)
    parser.add_argument('--num-classes', type=int, default=19)

    # optimizer
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--lr-type', type=str, default='poly')
    parser.add_argument('--base-lr', type=float, default=0.04)
    parser.add_argument('--lr-decay-step', type=int, default=40000)
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--wd', type=float, default=4.e-4)
    parser.add_argument('--loss-scale', type=float, default=1.)

    # model
    parser.add_argument('--model', type=str, default='eprnet')
    # parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')
    parser.add_argument('--ckpt-pretrained', type=str, default='')

    # train
    parser.add_argument('--device-target', type=str, choices=['GPU', 'CPU'], default='GPU')
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--group-size', type=int, default=1)
    parser.add_argument('--save-steps', type=int, default=32)
    parser.add_argument('--keep-checkpoint-max', type=int, default=5)

    return parser.parse_args()


def train():
    args = parse_args()

    # backend
    assert args.device_target == 'GPU'
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.distributed:
        init("nccl")
        args.rank = get_rank()
        args.group_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          device_num=args.group_size)
    # experiments directory
    args.train_dir = os.path.join(env_dir, args.train_dir, 'ckpt')
    if args.rank == 0:
        if os.path.exists(args.train_dir):
            shutil.rmtree(args.train_dir, ignore_errors=True)  # rm existing dir
        makedir_p(args.train_dir)
    args.data_file = os.path.join(env_dir, args.data_file)

    # dataset
    dataset = TransformSegDataset(data_file=args.data_file,
                                  batch_size=args.batch_size,
                                  crop_size=args.crop_size,
                                  min_scale=args.min_scale,
                                  max_scale=args.max_scale,
                                  ignore_label=args.ignore_label,
                                  num_classes=args.num_classes,
                                  shard_id=args.rank,
                                  shard_num=args.group_size)
    dataset = dataset.get_transformed_dataset(repeat=1)

    # network
    network = get_model_by_name(args.model, nclass=args.num_classes, phase='train')
    loss = SoftmaxCrossEntropyLoss(args.num_classes, ignore_label=args.ignore_label)
    loss.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss)

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * args.epochs
    lr_iter = lr_scheduler(lr_type=args.lr_type,
                           base_lr=args.base_lr,
                           total_train_steps=total_train_steps,
                           lr_decay_step=args.lr_decay_step,
                           lr_decay_rate=args.lr_decay_rate)
    opt = nn.Momentum(params=train_net.trainable_params(),
                      learning_rate=lr_iter,
                      momentum=args.momentum,
                      weight_decay=args.wd,
                      loss_scale=args.loss_scale)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    model = Model(train_net, optimizer=opt, amp_level='O0', loss_scale_manager=manager_loss_scale)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_steps,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory=args.train_dir, config=config_ck)
        cbs.append(ckpoint_cb)

    model.train(args.epochs, dataset, callbacks=cbs,
                dataset_sink_mode=(args.device_target != "CPU"))


if __name__ == '__main__':
    train()
