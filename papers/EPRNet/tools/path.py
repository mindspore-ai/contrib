# coding=utf-8

import os

__all__ = ['root_dir', 'makedir_p', 'dataset_dir', 'experiment_dir']


def root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def makedir_p(*paths):
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def dataset_dir():
    return os.path.join(root_dir(), 'dataset')


def _experiments_dir():
    return os.path.join(root_dir(), 'experiments')


def experiment_dir(model_name: str):
    return makedir_p(_experiments_dir(), model_name.lower())
