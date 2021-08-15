# coding=utf-8

import os
import sys
import shutil
import argparse

# &PATH
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)

from mindseg.data import build_data_file


def parse_args():
    parser = argparse.ArgumentParser('build mindrecord data')

    parser.add_argument('--data-name', type=str, default='Cityscapes')
    parser.add_argument('--dst-path', type=str, default='tmp_data/train.mindrecord',
                        help='relative path to root dir')
    parser.add_argument('--num-shard', type=int, default=4)
    parser.add_argument('--shuffle', action='store_true', default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # remove existing directory and make a new one
    args.dst_path = os.path.join(env_dir, args.dst_path)
    dir_name = os.path.dirname(args.dst_path)
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name)

    build_data_file(data_name=args.data_name,
                    split='train',
                    shard_num=args.num_shard,
                    shuffle=args.shuffle,
                    mindrecord_path=args.dst_path)
