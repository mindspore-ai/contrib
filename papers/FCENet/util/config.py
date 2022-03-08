from easydict import EasyDict
import os

config = EasyDict()

config.gpu = "1"

config.test_size=(640, 1280)
config.means=(0.485, 0.456, 0.406)
config.stds=(0.229, 0.224, 0.225)


# dataloader jobs number
config.num_workers = 0

# batch_size
config.batch_size = 2

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-2

# using GPU
config.cuda = True

# tr threshold
config.tr_thresh = 0.6
config.tcl_thresh = 0

# nms threshold
config.nms_thresh = 0.05

# k
config.k = 5

config.output_dir = 'output'

config.input_size = 640

# max polygon per image
config.max_annotation = 200

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
