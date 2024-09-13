"""
main.py
"""

import moxing as mox
from data import train_dataset, test_dataset
# from train import train, default_train
from train_ori0 import train

mox.file.shift('os', 'mox')

if __name__ == '__main__':

    psnrs, ssims = train(train_dataset, test_dataset)
    print("psnr:", psnrs)
    print("ssim:", ssims)
    # default_train(train_dataset, test_dataset)
