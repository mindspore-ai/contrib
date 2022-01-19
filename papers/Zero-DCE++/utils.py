"""
Utils for training.
"""

import logging
import os


class AverageUtil:
    """
    Util for dave the avg
    """

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        """
        reset the avg util
        """
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, value, *, count=1):
        """
        add new data into this utils
        """
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count


class Log:
    """
    logging util
    """
    def __init__(self, filename):
        assert not os.path.exists(filename), f"log file: {filename} exists!"
        logging.basicConfig(filename=filename, level=logging.INFO)

    def __call__(self, info):
        logging.info(info)
        print(info)
