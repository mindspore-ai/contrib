"""
For training
"""

import argparse
import os

from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.train.serialization import save_checkpoint

import dataset
import loss as L
import models
import utils

def train(config):
    """
    train function
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(save_graphs=False)

    dce_net = models.ZeroDCE()

    train_dataset = dataset.make_dataset(
        config.lowlight_images_path,
        config.batch_size,
        config.shuffle_size,
        image_type=config.image_type
    )

    loss = L.Loss(16, 0.6)

    optimizer = nn.Adam(
        dce_net.trainable_params(),
        learning_rate=config.lr,
        weight_decay=config.weight_decay
    )

    new_with_loss = WithLossCell(dce_net, loss)
    train_network = TrainOneStepCell(new_with_loss, optimizer)

    if config.pretrain_model is not None:
        param_dict = load_checkpoint(config.pretrain_model)
        load_param_into_net(train_network, param_dict)

    train_network.set_train()

    loss_avg = utils.AverageUtil()
    print_loss = utils.AverageUtil()
    min_avg_loss = 10000
    log = utils.Log(os.path.join(config.log_path, 'zero_dce.log'))
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
                            ckpt_file_name=f"{config.checkpoints_path}/zero_dce_{epoch}.ckpt")
        log(f"===== epoch {epoch}/{config.epochs} finished, mean_loss: {loss_avg.avg} =====")
        if loss_avg.avg < min_avg_loss:
            min_avg_loss = loss_avg.avg
            save_checkpoint(save_obj=dce_net,
                            ckpt_file_name=f"{config.checkpoints_path}/best_model.ckpt")
    save_checkpoint(save_obj=train_network,
                    ckpt_file_name="{config.checkpoints_path}/zero_dce_{config.epochs - 1}.ckpt")
    log("============== Stop Training ==============")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,
                        default="data/train_data/", help="the train data path")
    parser.add_argument('--image_type', type=str, default="jpg", help="the image postfix")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--shuffle_size', type=int, default=10, help="shuffle size")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--pretrain_model', type=str, default=None, help="the pretrain model path")
    parser.add_argument('--log_path', type=str, default='./', help="the log path")
    parser.add_argument('--epochs', type=int, default=200, help="epoch num")
    parser.add_argument('--iter_per_print', type=int,
                        default=10, help="how many iters to print the info")
    parser.add_argument('--epoch_per_save', type=int,
                        default=5, help="how many epooh to save the checkpoint")
    parser.add_argument('--checkpoints_path', type=str,
                        default='./checkpoints', help="the checkpoint path")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    train(args)
