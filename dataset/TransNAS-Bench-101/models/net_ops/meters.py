# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""meters"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__)


class RecorderMeter:
    """RecorderMeter"""
    def __init__(self, labels):
        self.labels = labels
        self.num_metrics = len(self.labels)
        self.metrics = {x: [] for x in self.labels}

    def reset(self):
        """reset"""
        self.metrics = {x: [] for x in self.labels}

    def update(self, metric_dict):
        """update"""
        for key, value in metric_dict.items():
            try:
                self.metrics[key].append(value)
            except ValueError:
                print("ValueError")

    def max_metric(self, metric):
        """max_metric"""
        return max(self.metrics[metric])

    def load_from_model_db(self, metrics):
        """load_from_model_db"""
        print("loading from model_db.json")
        epoch_xs = [f"epoch_{x}" for x in range(len(metrics.keys()))]
        for epoch_x in epoch_xs:
            self.update(metrics[epoch_x])

    def plot_curve(self, save_path):
        """plot_curve"""
        matplotlib.use('agg')
        title = 'the metrics/loss curve of train/val/test'
        dpi = 100
        width, height = 1600, 1000
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)

        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)

        arrs, labels = [], []
        for key, value in self.metrics.items():
            if 'loss' in key or 'ssim' in key:
                arrs.append([i * 20 for i in value])
                labels.append(key + '-x20')
            elif 'top5' in key:
                continue
            else:
                arrs.append(value)
                labels.append(key)

        colors = ['g', 'y', 'b']
        linestyles = ['-', ':', '--', '-.']
        label_modes = list(set(la.split("_")[0] for la in labels))
        label_types = list(set('_'.join(la.split("_")[1:]) for la in labels))
        x = list(range(len(arrs[0])))
        for _, (label, arr) in enumerate(zip(labels, arrs)):
            la_mode, la_type = label.split("_")[0], '_'.join(label.split("_")[1:])
            color, linestyle = colors[label_modes.index(la_mode)], linestyles[label_types.index(la_type)]
            plt.plot(x, arr, color=color, linestyle=linestyle, label=label, lw=2)

        if 'valid_top1' in labels:
            annot_max(x, self.metrics['valid_top1'])
        elif 'valid_mIoU' in labels:
            annot_max(x, self.metrics['valid_mIoU'])
        elif 'val_ssim' in labels:
            annot_max(x, self.metrics['val_ssim'])

        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def annot_max(x, y, ax=None):
    """annot_max"""
    x, y = np.array(x), np.array(y)
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="-")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)
