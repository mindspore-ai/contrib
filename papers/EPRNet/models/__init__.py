# coding=utf-8

from .eprnet import *

_nets_map = {
    'eprnet': EPRNet,
}


def get_model_by_name(name: str, **kwargs):
    return _nets_map[name.lower()](**kwargs)
