# coding=utf-8
# adapted from:
# https://github.com/mindspore-ai/mindspore/blob/master/model_zoo/official/cv/deeplabv3/eval.py

import os
import cv2
import sys
import time
import argparse
import numpy as np
from mindspore import nn, context, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# &PATH
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)

from mindseg.data import get_files_list
from mindseg.models import get_model_by_name
from mindseg.tools import validate_ckpt


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network

        self.transpose = P.Transpose()
        self.softmax = nn.Softmax(axis=3)  # support only for calculation at the lase axis for now

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.transpose(output, (0, 2, 3, 1))  # tranpose from NCHW to NHWC
        output = self.softmax(output)
        output = self.transpose(output, (0, 3, 1, 2))  # transpose NHWC to NCHW
        return output


def parse_args():
    parser = argparse.ArgumentParser('mindspore semantic segmentation evaluation')

    # val data
    parser.add_argument('--data-name', type=str, default='cityscapes')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--scales', type=float, nargs='+', default=(1.0,))
    # parser.add_argument('--scales', type=float, nargs='+', default=(0.5, 0.75, 1.0, 1.25, 1.75, 2.0))
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--ignore-label', type=int, default=19)
    parser.add_argument('--num-classes', type=int, default=19)
    parser.add_argument('--eval-split', type=str, default='val')

    # model
    parser.add_argument('--model', type=str, default='eprnet')
    parser.add_argument('--checkpoint', type=str, default='experiment/ckpt/eprnet-10_27.ckpt',
                        help='relative path to root dir')

    # speed
    parser.add_argument('--speed', action='store_true', default=False)
    parser.add_argument('--data-size', type=int, nargs='+', default=(1024, 2048))

    return parser.parse_args()


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def pre_process(img_, crop_size=513,
                img_mean=(103.53, 116.28, 123.675),
                img_std=(57.375, 57.120, 58.395)):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(img_mean)
    image_std = np.array(img_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def eval_batch(eval_net, img_lst, crop_size=513, flip=True):
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def eval_batch_scales(eval_net, img_lst, scales, base_crop_size=513, flip=True):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def evaluation(args):
    # data list
    images_list, masks_list = get_files_list(args.data_name, split=args.eval_split)

    # model
    network = get_model_by_name(args.model, nclass=args.num_classes, phase='eval')
    eval_net = BuildEvalNetwork(network)
    load_param_into_net(eval_net, load_checkpoint(validate_ckpt(env_dir, args.checkpoint)))
    eval_net.set_train(False)

    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    image_num = 0
    for idx in range(len(images_list)):
        img_path = images_list[idx]
        msk_path = masks_list[idx]
        img_ = cv2.imread(img_path)
        msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size, flip=args.flip)
            for mi in range(args.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            print('processed {} images'.format(idx + 1))
        image_num = idx

    if bi > 0:
        batch_res = eval_batch_scales(eval_net, batch_img_lst, scales=args.scales,
                                      base_crop_size=args.crop_size, flip=args.flip)
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        print('processed {} images'.format(image_num + 1))

    # print(hist)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('per-class IoU', iu)
    print('mean IoU', np.nanmean(iu))


def _sample(shape) -> Tensor:
    if isinstance(shape, (list, tuple)):
        h = shape[0]
        w = shape[1]
    else:
        h = shape
        w = shape
    sample = Tensor(np.random.uniform(size=(1, 3, h, w)), mstype.float32)
    return sample


def speed(model_name, nclass, data_size=(1024, 2048), iterations=1000, warm_up=500):
    network = get_model_by_name(model_name, nclass=nclass, phase='eval')
    network.set_train(False)
    sample = _sample(data_size)

    # warm-up
    print(f'Warm-up starts for {warm_up} forward passes...')
    for _ in range(warm_up):
        network(sample)

    print(f'Evaluate inference speed for {iterations} forward passes...')
    start = time.time()
    for _ in range(iterations):
        network(sample)
    time_cost = time.time() - start

    print('Total time: %.2fs, latency: %.2fms, FPS: %.1f'
          % (time_cost, time_cost / iterations * 1000, iterations / time_cost))


if __name__ == '__main__':
    eval_args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    if not eval_args.speed:
        evaluation(eval_args)
    else:
        speed(model_name=eval_args.model,
              nclass=eval_args.num_classes,
              data_size=eval_args.data_size,
              iterations=1000,
              warm_up=500)
