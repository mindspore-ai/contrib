#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""LIE-IQA-Net"""

import os
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
import mindspore.numpy as msnp
from mindspore import Parameter
from mindspore import load_checkpoint, load_param_into_net
from networks.vgg import vgg16_feats


class DecomNet(nn.Cell):
    """DecomNet"""
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4,
                                    channel,
                                    kernel_size * 3,
                                    pad_mode='pad',
                                    padding=4,
                                    weight_init='normal',
                                    has_bias=True)
        self.net1_convs = nn.SequentialCell([
            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1, weight_init='normal', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1, weight_init='normal', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1, weight_init='normal', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1, weight_init='normal', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1, weight_init='normal', has_bias=True),
            nn.ReLU()
        ])
        self.net1_recon = nn.Conv2d(channel,
                                    4,
                                    kernel_size,
                                    pad_mode='pad',
                                    padding=1,
                                    weight_init='normal',
                                    has_bias=True)
        self.sigmoid = nn.Sigmoid()

    def construct(self, input_im):
        """build"""
        input_max = msnp.amax(input_im, axis=1, keepdims=True)
        cat = ops.Concat(axis=1)
        input_img = cat((input_max, input_im))
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        reflectance = self.sigmoid(outs[:, 0:3, :, :])
        luminance = self.sigmoid(outs[:, 3:4, :, :])
        return reflectance, luminance


class L2pooling(nn.Cell):
    """L2pooling"""
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        self.filter_size = filter_size
        a = np.hanning(filter_size)[1:-1]  # hanning window
        g = Tensor(a[:, None] * a[None, :], mindspore.float32)
        reduce_sum = ops.ReduceSum()
        g = g / reduce_sum(g)
        tile = ops.Tile()
        self.filter = Parameter(tile(g[None, None, :, :], (self.channels, 1, 1, 1)))

    def construct(self, x):
        """build"""
        x = x ** 2
        conv2d = ops.Conv2D(out_channel=self.channels,
                            kernel_size=self.filter.shape[-1],
                            pad_mode='pad',
                            pad=self.padding,
                            stride=self.stride,
                            group=x.shape[1])
        out = conv2d(x, self.filter)
        ops_sqrt = ops.Sqrt()
        out = ops_sqrt(out + 1e-12)
        return out


class RetinexLieIqaNetVgg(nn.Cell):
    """RetinexLieIqaNetVgg"""
    def __init__(self, load_weight=False):
        super(RetinexLieIqaNetVgg, self).__init__()
        vgg_features = vgg16_feats()
        self.stage1 = nn.SequentialCell([vgg_features.layers[idx] for idx in range(0, 4)])
        self.stage2_l2pool = L2pooling(channels=64)
        self.stage2 = nn.SequentialCell([vgg_features.layers[idx] for idx in range(5, 9)])
        self.stage3_l2pool = L2pooling(channels=128)
        self.stage3 = nn.SequentialCell([vgg_features.layers[idx] for idx in range(10, 16)])
        self.stage4_l2pool = L2pooling(channels=256)
        self.stage4 = nn.SequentialCell([vgg_features.layers[idx] for idx in range(17, 23)])
        self.stage5_l2pool = L2pooling(channels=512)
        self.stage5 = nn.SequentialCell([vgg_features.layers[idx] for idx in range(24, 30)])
        for name_param in self.parameters_and_names():
            name_param[1].requires_grad = False
        self.feats_chns = [3, 64, 128, 256, 512, 512]
        alpha = Tensor(np.random.normal(0.1, 0.01, (1, sum(self.feats_chns), 1, 1)), dtype=mindspore.float32)
        beta = Tensor(np.random.normal(0.1, 0.01, (1, sum(self.feats_chns), 1, 1)), dtype=mindspore.float32)
        gamma = Tensor(np.random.normal(0.1, 0.01, (1, sum(self.feats_chns), 1, 1)), dtype=mindspore.float32)
        self.gamma = Parameter(gamma, name='gamma', requires_grad=True)
        self.alpha = Parameter(alpha, name='alpha', requires_grad=True)
        self.beta = Parameter(beta, name='beta', requires_grad=True)
        feat_mean = Tensor([0.485, 0.456, 0.406], dtype=mindspore.float32).view(1, -1, 1, 1)
        feat_std = Tensor([0.229, 0.224, 0.225], dtype=mindspore.float32).view(1, -1, 1, 1)
        self.mean = Parameter(feat_mean, name='mean', requires_grad=False)
        self.std = Parameter(feat_std, name='std', requires_grad=False)
        self.decom_net = DecomNet()
        for name_param in self.decom_net.parameters_and_names():
            name_param[1].requires_grad = False
        if load_weight:
            root_dir = os.path.dirname(os.path.realpath(__file__))
            weight_dict = load_checkpoint(os.path.join(root_dir, '..', 'checkpoint/Retinex_LIEIQANet.ckpt'))
            param_not_load = load_param_into_net(self, weight_dict)
            print('parameters not load to net: ', param_not_load)

    def construct_once(self, x):
        """construct_once"""
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2_l2pool(h)
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3_l2pool(h)
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4_l2pool(h)
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5_l2pool(h)
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def feat_distance(self, x_feats, y_feats, weights):
        """feat_distance"""
        c0 = 1e-4
        dist = 0.
        ms_sqrt = ops.Sqrt()
        ms_sum = ops.ReduceSum(keep_dims=True)
        for idx in range(len(self.feats_chns)):
            x_mean = x_feats[idx].mean([2, 3], keep_dims=True)
            y_mean = y_feats[idx].mean([2, 3], keep_dims=True)
            x_var = ((x_feats[idx] - x_mean) ** 2).mean([2, 3], keep_dims=True)
            y_var = ((y_feats[idx] - y_mean) ** 2).mean([2, 3], keep_dims=True)
            x_feats[idx] = ((x_feats[idx] - x_mean) + c0) / (ms_sqrt(x_var) + c0)
            y_feats[idx] = ((y_feats[idx] - y_mean) + c0) / (ms_sqrt(y_var) + c0)
            s0 = ((x_feats[idx] - y_feats[idx]) ** 2).mean([2, 3], keep_dims=True)
            dist = dist + ms_sum(weights[idx] * s0, axis=1)
        return dist

    def construct(self, x, y, batch_average=False):
        """build"""
        _, x_c, _, _ = x.shape
        _, y_c, _, _ = y.shape
        assert x_c == 3 and y_c == 3, 'the channel of input must be 3'
        x_r, x_s = self.decom_net(x)
        y_r, y_s = self.decom_net(y)
        ms_concat = ops.Concat(axis=1)
        x_s = ms_concat([x_s, x_s, x_s])
        y_s = ms_concat([y_s, y_s, y_s])
        assert x_r.shape == y_r.shape, 'shape of x and y must the same'
        x_feats = self.construct_once(x)
        y_feats = self.construct_once(y)
        r_feats1 = self.construct_once(x_r)
        r_feats2 = self.construct_once(y_r)
        s_feats1 = self.construct_once(x_s)
        s_feats2 = self.construct_once(y_s)
        ops_sum = ops.ReduceSum()
        split_chns = [3, 67, 195, 451, 963, 1475]
        w_sum = ops_sum(self.alpha) + ops_sum(self.beta) + ops_sum(self.gamma)
        alpha = msnp.split(self.alpha / w_sum, split_chns, axis=1)
        beta = msnp.split(self.beta / w_sum, split_chns, axis=1)
        gamma = msnp.split(self.gamma / w_sum, split_chns, axis=1)
        # input image
        dist = self.feat_distance(x_feats, y_feats, gamma)
        # reflectance
        r_dist = self.feat_distance(r_feats1, r_feats2, alpha)
        # shading
        s_dist = self.feat_distance(s_feats1, s_feats2, beta)
        # larger is worse
        score = (dist + r_dist + s_dist).squeeze()
        if batch_average:
            return score.mean()
        return score


if __name__ == '__main__':
    input_x = Tensor(np.ones([1, 3, 256, 256]), mindspore.float32)
    input_y = Tensor(np.ones([1, 3, 256, 256]), mindspore.float32)
    model = RetinexLieIqaNetVgg(load_weight=True)
    pred = model(input_x, input_y)
    print(pred)
