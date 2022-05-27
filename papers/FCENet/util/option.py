import argparse
import os

from datetime import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('--exp_name', default="Icdar2015", type=str,
                                 choices=['Totaltext', 'Ctw1500', 'Icdar2015'], help='Experiment name')
        self.parser.add_argument("--k", default= 5, help="Maximum Fourier Coefficient", type=int)
        self.parser.add_argument("--gpu", default="1", help="set gpu id", type=str)
        self.parser.add_argument('--resume', default="convert_from_torch1.ckpt", type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--mgpu', action='store_true', help='Use multi-gpu to train model')
        self.parser.add_argument('--save_dir', default='./model/', help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default='./vis/', help='Path to save visualization images')
        self.parser.add_argument('--log_dir', default='./logs/', help='Path to tensorboard log')

        self.parser.add_argument('--verbose', '-v', default=True, type=str2bool, help='Whether to output debug info')
        self.parser.add_argument('--viz', action='store_true', help='Whether to output debug info')
        self.parser.add_argument('--dcn', action='store_true', help='Whether to use DCN')

        # train opts
        self.parser.add_argument('--max_epoch', default=200, type=int, help='Max epochs')
        self.parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
        self.parser.add_argument('--weight_decay', '--wd', default=0., type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--save_freq', default=5, type=int, help='save weights every # epoch')
        self.parser.add_argument('--display_freq', default=20, type=int, help='display training metrics every # iter')
        self.parser.add_argument('--viz_freq', default=20, type=int, help='visualize training process every # iter')
        self.parser.add_argument('--log_freq', default=100, type=int, help='log to tensorboard every # iterations')
        self.parser.add_argument('--val_freq', default=100, type=int, help='do validation every # iterations')
        self.parser.add_argument('--input_size', default=800, type=int, help='model input size')

        # data args
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--means', type=int, default=(0.485, 0.456, 0.406), nargs='+', help='mean')
        self.parser.add_argument('--stds', type=int, default=(0.229, 0.224, 0.225), nargs='+', help='std')

        # eval and demo args
        self.parser.add_argument('--tr_thresh', default=0.75, type=float, help='tr')
        self.parser.add_argument('--nms_thresh', default=0.1, type=float, help='nms_thresh')
        self.parser.add_argument('--test_size', default=(640, 1280), type=int, nargs='+',
                                 help='model input size')  # ctw1500: (640, 1280) totaltext: (960, 1280)
        self.parser.add_argument('--test_root', default=None, type=str, help='Path to test images')


    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
