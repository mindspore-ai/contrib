"""
for testing
"""

import argparse
import os.path

import numpy as np
from PIL import Image
from mindspore import load_checkpoint, load_param_into_net, context

import dataset
import models


def tensor2image(tensor):
    """
    transfer tensor to numpy
    """
    img = tensor.asnumpy()
    img *= 255
    img = img.clip(0, 255)
    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))
    return img


def test(config):
    """
    test by the config.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(save_graphs=False)

    dce_net = models.ZeroDCEpp(scale_factor=1.0)
    dce_net.set_grad(False)
    test_dataset = dataset.make_test_dataset(
        config.lowlight_images_path,
        config.batch_size,
        image_type=config.image_type
    )

    param_dict = load_checkpoint(config.pretrain_model)
    load_param_into_net(dce_net, param_dict)

    count = 0
    for i, data in enumerate(test_dataset):
        print(
            f"processing the {i * config.batch_size}-{(i + 1) * config.batch_size} image")
        y, _ = dce_net(data[0])
        index = 0
        for img in y:
            img = tensor2image(img)
            if config.save_with_src:
                src_img = tensor2image(data[0][index, :, :, :])
                img = np.concatenate((src_img, img), axis=1)
            image = Image.fromarray(img)
            image.save(os.path.join(config.save_path, "{count}.jpg"))
            count += 1
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,
                        default="data/test_data/real/", help="the test data path.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--pretrain_model', type=str,
                        default='./pretrain_model/zero_dcepp_epoch99.ckpt',
                        help="the pretrain model path")
    parser.add_argument('--save_with_src', action='store_true',
                        help="whether save the source image")
    parser.add_argument('--save_path', type=str,
                        default='./outputs/real', help="the output dir")
    parser.add_argument('--image_type', type=str,
                        default="png", help="the image postfix")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    test(args)
