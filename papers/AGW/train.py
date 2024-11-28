# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================
'''train net'''

import warnings
import os
import os.path as osp
import numpy as np

import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import Tensor, Model, context
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import (Callback, ModelCheckpoint, CheckpointConfig, LossMonitor,
                                      TimeMonitor, SummaryCollector)
from mindspore.communication.management import init

from eval import do_eval
from src.utils.loss import OriTripletLoss, TripletLossWRT, CrossEntropyLoss
from src.agw import AGWLoss, create_agw_net
from src.utils.local_adapter import get_device_id, get_device_num
from src.utils.config import config
from src.data.dataset import dataset_creator
from src.utils.lr_generator import step_lr, multi_step_lr, warmup_step_lr


set_seed(1)


class LossCallBack(LossMonitor):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        '''check loss at the end of each step.'''
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num -
                             1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss), flush=True)


class EvalCallBack(Callback):
    '''
    Train-time Evaluation
    '''

    def __init__(self, net, eval_per_epoch):
        self.net = net
        self.eval_per_epoch = eval_per_epoch

        _, self.query_dataset = dataset_creator(
            root=config.data_path, height=config.height, width=config.width,
            dataset=config.target, norm_mean=config.norm_mean,
            norm_std=config.norm_std, batch_size_test=config.batch_size_test,
            workers=config.workers, cuhk03_labeled=config.cuhk03_labeled,
            cuhk03_classic_split=config.cuhk03_classic_split,
            mode='query')
        _, self.gallery_dataset = dataset_creator(
            root=config.data_path, height=config.height,
            width=config.width, dataset=config.target,
            norm_mean=config.norm_mean, norm_std=config.norm_std,
            batch_size_test=config.batch_size_test, workers=config.workers,
            cuhk03_labeled=config.cuhk03_labeled,
            cuhk03_classic_split=config.cuhk03_classic_split,
            mode='gallery')

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            do_eval(self.net, self.query_dataset, self.gallery_dataset)


def init_lr(num_batches):
    '''initialize learning rate.'''
    if config.lr_scheduler == 'single_step':
        lr = step_lr(config.lr, config.step_epoch, num_batches,
                     config.max_epoch, config.gamma)
    elif config.lr_scheduler == 'multi_step':
        lr = multi_step_lr(config.lr, config.step_epoch,
                           num_batches, config.max_epoch, config.gamma)
    elif config.lr_scheduler == 'cosine':
        lr = np.array(nn.cosine_decay_lr(0., config.lr, num_batches * config.max_epoch, num_batches,
                                         config.max_epoch)).astype(np.float32)
    elif config.lr_scheduler == 'warmup_step':
        lr = warmup_step_lr(config.lr, config.step_epoch, num_batches, config.warmup_epoch,
                            config.max_epoch, config.gamma)

    return lr


def check_isfile(fpath):
    '''check whether the path is a file.'''
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def load_from_checkpoint(net):
    '''load parameters when resuming from a checkpoint for training.'''
    param_dict = load_checkpoint(config.checkpoint_file_path)
    if param_dict:
        if param_dict.get("epoch_num") and param_dict.get("step_num"):
            config.start_epoch = int(param_dict["epoch_num"].data.asnumpy())
            config.start_step = int(param_dict["step_num"].data.asnumpy())
        else:
            config.start_epoch = 0
            config.start_step = 0
        load_param_into_net(net, param_dict)
    else:
        raise ValueError("Checkpoint file:{} is none.".format(
            config.checkpoint_file_path))


def set_save_ckpt_dir():
    """set save ckpt dir"""
    ckpt_save_dir = os.path.join(
        config.output_path, config.checkpoint_path, config.source)

    return ckpt_save_dir


def get_callbacks(num_batches):
    '''get all callback list'''
    time_cb = TimeMonitor(data_size=num_batches)
    loss_cb = LossCallBack()
    summary_collector = SummaryCollector(
        summary_dir='./summary_dir', collect_freq=1)

    cb = [time_cb, loss_cb, summary_collector]

    ckpt_append_info = [
        {"epoch_num": config.start_epoch, "step_num": config.start_epoch}]
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * num_batches,
                                 keep_checkpoint_max=5, append_info=ckpt_append_info)
    ckpt_cb = ModelCheckpoint(
        prefix="agw-0321", directory=set_save_ckpt_dir(), config=config_ck)
    cb += [ckpt_cb]

    return cb


def train_net():
    """train net"""
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)
    device_num = get_device_num()
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel',
                                              gradients_mean=True)
            init()

    num_classes, dataset1 = dataset_creator(
        root=config.data_path, height=config.height, width=config.width,
        transforms=config.transforms, dataset=config.source,
        norm_mean=config.norm_mean, norm_std=config.norm_std,
        batch_size_train=config.batch_size_train, workers=config.workers,
        cuhk03_labeled=config.cuhk03_labeled,
        cuhk03_classic_split=config.cuhk03_classic_split, mode='train')

    num_batches = dataset1.get_dataset_size()

    # =================================

    net = create_agw_net(num_class=num_classes, last_stride=config.last_stride)

    agwloss = AGWLoss(
        ce=CrossEntropyLoss(num_classes=num_classes,
                            label_smooth=config.label_smooth),
        tri=TripletLossWRT() if config.wrt_loss else OriTripletLoss(
            batch_size=dataset1.get_batch_size())
    )

    lr = init_lr(num_batches=num_batches)

    if config.optim == 'adam':
        opt2 = nn.Adam(net.trainable_params(), learning_rate=lr,
                       beta1=config.adam_beta1, beta2=config.adam_beta2, weight_decay=config.weight_decay)
    else:
        opt2 = nn.Momentum(net.trainable_params(), learning_rate=lr,
                           momentum=config.momentum, weight_decay=config.weight_decay, use_nesterov=True)

    model2 = Model(network=net, optimizer=opt2, loss_fn=agwloss)

    callbacks = get_callbacks(num_batches)
    callbacks += [EvalCallBack(net, 20)]

    model2.train(config.max_epoch, dataset1, callbacks, dataset_sink_mode=True)

    print("======== Train success ========")


if __name__ == '__main__':
    train_net()
