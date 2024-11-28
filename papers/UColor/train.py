"""
for training
"""

import argparse
import os

import mindspore.ops.operations as P
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn
from mindspore.nn import WithLossCell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.train.serialization import save_checkpoint

import utils
from datasets import make_dataset
from loss import Loss
from model import Ucolor

GRADIENT_CLIP_TYPE = 0
VGG_CKPT_FILE = './pretrain_vgg/vgg16_ascend_v130_imagenet2012'\
                '_official_cv_bs64_top1acc73.49__top5acc91.56.ckpt'

class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """Defines the gradients clip."""
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(
                    F.tuple_to_array((clip_value,)), dt))
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads


class ClipTrainOneStepCell(nn.TrainOneStepCell):
    """
    Encapsulation class of GRU network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        enable_clip_grad (boolean): If True, clip gradients in ClipTrainOneStepCell. Default: True.
    """

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True, grad_clip_norm=0.1):
        super(ClipTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()
        self.clip_gradients = ClipGradients()
        self.enable_clip_grad = enable_clip_grad
        self.grad_clip_norm = grad_clip_norm

    def set_sens(self, value):
        """set sens"""
        self.sens = value

    def construct(self, x, y):
        """Defines the computation performed."""

        weights = self.weights
        loss = self.network(x, y)

        grads = self.grad(self.network, weights)(x, y, self.cast(F.tuple_to_array((self.sens,)),
                                                                 mstype.float32))
        if self.enable_clip_grad:
            grads = self.clip_gradients(
                grads, GRADIENT_CLIP_TYPE, self.grad_clip_norm)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


def train(config):
    """train main function"""
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(save_graphs=False)

    ucolor = Ucolor()

    train_dataset = make_dataset(input_path=config.input_path,
                                 depth_path=config.depth_path,
                                 gt_path=config.gt_path,
                                 type_="train",
                                 image_type=config.image_type,
                                 patch_size=config.patch_size,
                                 transform=None,
                                 batch_size=config.batch_size,
                                 shuffle_size=config.shuffle_size,
                                 num_parallel_workers=config.num_parallel_workers)

    loss = Loss(config.vgg_pretrain_file)

    optimizer = nn.Adam(
        ucolor.trainable_params(),
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        beta1=config.beta1
    )
    new_with_loss = WithLossCell(ucolor, loss)
    train_network = ClipTrainOneStepCell(
        new_with_loss, optimizer, grad_clip_norm=config.grad_clip_norm)

    if config.pretrain_model is not None:
        param_dict = load_checkpoint(config.pretrain_model)
        load_param_into_net(train_network, param_dict)

    train_network.set_train()

    loss_avg = utils.AverageUtil()
    print_loss = utils.AverageUtil()
    min_avg_loss = 1000000
    log = utils.Log(os.path.join(config.log_path, 'ucolor.log'))
    log("============== Start Training ==============")
    for epoch in range(config.epochs):
        loss_avg.reset()
        print_loss.reset()
        for idx, data in enumerate(train_dataset):
            loss = train_network(*data)
            loss_avg.update(loss)
            print_loss.update(loss)
            if print_loss.count == config.iter_per_print:
                log(f"loss: {loss}, iter: {idx}, epoch: {epoch}/{config.epochs}")
                print_loss.reset()
        if epoch % config.epoch_per_save == 0:
            save_checkpoint(save_obj=train_network,
                            ckpt_file_name=f"{config.checkpoints_path}/ucolor_{epoch}.ckpt")
        log(f"===== epoch {epoch}/{config.epochs} finished, mean_loss: {loss_avg.avg} =====")
        if loss_avg.avg < min_avg_loss:
            min_avg_loss = loss_avg.avg
            save_checkpoint(
                save_obj=ucolor, ckpt_file_name=f"{config.checkpoints_path}/best_model.ckpt")
    save_checkpoint(save_obj=train_network,
                    ckpt_file_name=f"{config.checkpoints_path}/ucolor_{config.epochs - 1}.ckpt")
    log("============== Stop Training ==============")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--input_path', type=str,
                        default="data/input_train/", help="the train data path")
    parser.add_argument('--depth_path', type=str,
                        default="data/depth_train/", help="the depth data path")
    parser.add_argument('--gt_path', type=str,
                        default="data/gt_train/", help="the gt data path")
    parser.add_argument('--image_type', type=str,
                        default="png", help="the image postfix")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--patch_size', type=int,
                        default=128, help="patch size")
    parser.add_argument('--shuffle_size', type=int,
                        default=10, help="shuffle size")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--beta1', type=float, default=0.5, help="beta1")
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help="weight decay")
    parser.add_argument('--pretrain_model', type=str,
                        default=None, help="the pretrain model path")
    parser.add_argument('--log_path', type=str,
                        default='./', help="the log path")
    parser.add_argument('--epochs', type=int, default=200, help="epoch num")
    parser.add_argument('--iter_per_print', type=int, default=1,
                        help="how many iters to print the info")
    parser.add_argument('--epoch_per_save', type=int, default=1,
                        help="how many epoch to save the checkpoint")
    parser.add_argument('--checkpoints_path', type=str,
                        default='./checkpoints', help="the checkpoint path")
    parser.add_argument('--num_parallel_workers', type=int,
                        default=8, help="workers")
    parser.add_argument('--grad_clip_norm', type=float,
                        default=0.1, help="clip the grad")
    parser.add_argument('--vgg_pretrain_file', type=str,
                        default=VGG_CKPT_FILE,
                        help="the checkpoint path")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    train(args)
