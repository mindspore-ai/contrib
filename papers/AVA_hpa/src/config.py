"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed
import time
import json
import logging
import os


def get_pretrain_config():
    time_prefix = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    prefix = "AVA-hpa-pretrain"
    config = ed({
        # base setting
        "description": "this is the description for currnet config file.",
        "prefix": prefix,
        "time_prefix": time_prefix,
        "network": "resnet18",
        "low_dims": 128,
        "use_MLP": True,

        # save
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 2,

        # dataset
        "dataset": "hpa",
        "bag_size": 1,
        "classes": 10,
        "num_parallel_workers": 8,

        # optimizer
        "base_lr": 0.003,
        "type": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "loss_scale": 1,
        "sigma": 0.1,

        # trainer
        "breakpoint_training_path": "",
        "batch_size": 32,
        "epochs": 100,
        "lr_schedule": "cosine_lr",
        "lr_mode": "epoch",
        "warmup_epoch": 0,
    })
    return config


def get_train_config():
    time_prefix = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    prefix = "AVA-hpa-train"
    config = ed({
        # base setting
        "description": "this is the description for currnet config file.",
        "prefix": prefix,
        "time_prefix": time_prefix,
        "network": "resnet18",
        "low_dims": 128,
        "use_MLP": False,

        # save
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 2,

        # dataset
        "dataset": "hpa",
        "bag_size_for_train": 1,
        "bag_size_for_eval": 20,
        "classes": 10,
        "num_parallel_workers": 8,

        # optimizer
        "base_lr": 0.0001,
        "type": "Adam",
        "beta1": 0.5,
        "beta2": 0.999,
        "weight_decay": 0,
        "loss_scale": 1,

        # trainer
        "breakpoint_training_path": "",
        "batch_size_for_train": 8,
        "batch_size_for_eval": 1,
        "epochs": 20,
        "eval_per_epoch": 1,
        "lr_schedule": "cosine_lr",
        "lr_mode": "epoch",
        "warmup_epoch": 0,
    })
    return config


def save_config(paths, config, args_opt):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        file = open(path, "w")
        dicts = dict(config, **args_opt)
        json.dump(dicts, file, indent=4)
        file.close()


def get_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w+")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
