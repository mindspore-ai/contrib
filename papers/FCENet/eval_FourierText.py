import os
import sys
import time
import cv2
import numpy as np
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net, Model, nn, Tensor, context
from dataset import TotalText, Ctw1500Text, Icdar15Text
from network.textnet import TextNet
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.visualize import visualize_detection, visualize_gt
from util.misc import mkdirs, rescale_result
from util.detector import TextDetector
from util.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, data_transfer_ICDAR
from util.iterable import MyIterable
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file should be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(detector, test_loader, output_dir, testset, length):
    total_time = 0.
    idx = 0  # test mode can only run with batch_size == 1
    if not os.path.exists(output_dir):
        mkdirs(output_dir)

    for i, data in enumerate(test_loader):
        image = data['image']
        label_3 = data['label_3']
        label_4 = data['label_4']
        label_5 = data['label_5']
        meta = testset.get_meta_item(i)
        start = time.time()

        # visualization
        img_show = np.transpose(image[idx], (1,2,0))
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        # get detection result
        image_detect = Tensor(image)
        contours, output = detector.detect(image_detect)

        end = time.time()
        total_time += end - start
        fps = (i + 1) / total_time
        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, length, meta['image_id'], fps))

        cls_predict = output['tr']
        cls_gt = [(label_3[0, :, :, 0]*label_3[0, :, :, -1]),
                  (label_4[0, :, :, 0]*label_4[0, :, :, -1]),
                  (label_5[0, :, :, 0]*label_5[0, :, :, -1])]

        pred_vis = visualize_detection(img_show, contours, cls_predict)

        gt_contour = []
        for annot, n_annot in zip(meta['annotation'], meta['n_annotation']):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].astype(int))
        gt_vis = visualize_gt(img_show, gt_contour, cls_gt)

        im_vis = np.concatenate([pred_vis, gt_vis], axis=0)
        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'])
        cv2.imwrite(path, im_vis)

        H, W = meta['Height'], meta['Width']
        img_show, contours = rescale_result(img_show, contours, H, W)

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        else:
            fname = meta['image_id'].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))
        


def main(vis_dir_path):

    osmkdir(vis_dir_path)
    if cfg.exp_name == "Totaltext":
        testset = TotalText(
            data_root='data/total-text-mat',
            k=cfg.k,
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )

    elif cfg.exp_name == "Ctw1500":
        testset = Ctw1500Text(
            data_root='data/ctw1500',
            k=cfg.k,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Icdar2015":
        testset = Icdar15Text(
            data_root='data/icdar2015',
            k=cfg.k,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        print("{} is not justify".format(cfg.exp_name))
        sys.exit(0)

    testsetload = MyIterable(dataset=testset)
    testset_len = testsetload.len
    test_loader = ds.GeneratorDataset(testsetload,
                                      ['image', 'label_3', 'label_4', 'label_5'],
                                      shuffle=False)

    test_loader = test_loader.batch(batch_size=1)

    # Model
    model = TextNet(k=cfg.k,
                    dcn=False,
                    is_training=False)

    model_path = cfg.resume
    print('Loading from {}'.format(model_path))
    param_dict = load_checkpoint(model_path)
    load_param_into_net(model, param_dict, strict_load=True)

    detector = TextDetector(model)

    print('Start testing FourierText.')
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    test_dataset = test_loader.create_dict_iterator(output_numpy=True)
    inference(detector, test_dataset, output_dir, testsetload, testset_len)

    if cfg.exp_name == "Totaltext":
        deal_eval_total_text(debug=True)

    elif cfg.exp_name == "Ctw1500":
        deal_eval_ctw1500(debug=True)

    elif cfg.exp_name == "Icdar2015":
        deal_eval_icdar15(debug=True)

    else:
        print("{} is not justify".format(cfg.exp_name))


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)
