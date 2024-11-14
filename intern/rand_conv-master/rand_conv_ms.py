import mindspore
from mindspore import nn, context, Tensor, Parameter
import mindspore.ops as ops
from mindspore.common.initializer import Normal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import fire
import pprint

def do_rand_conv(kernel_size=3,
                 weight_init='normal',
                 alpha=0.7,
                 save_images=False):



    print('运行函数，参数设置如下：')
    pprint.pprint(locals(), width=1)
    print('....')

    img_path = 'images/robot.jpg'
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_float = image.astype('float32') / 255.0

    m = nn.Conv2d(3, 3, kernel_size, stride=1, pad_mode='pad', padding=kernel_size//2, has_bias=False)

    input_im = Tensor(image, mindspore.float32)
    input_im = ops.Transpose()(input_im, (2, 0, 1))  
    input_im = input_im.expand_dims(0)  
    input_im = input_im / 255.0

    fig, ax = plt.subplots(1)
    im1 = ax.imshow(image_float, extent=(0, 1, 1, 0))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for i in range(0, 1000):
        if weight_init == 'normal':
            std_normal = 1 / (np.sqrt(3) * kernel_size)
        elif weight_init == 'xavier':
            std_normal = 1 / np.sqrt(3)

        weight_shape = m.weight.shape
        weights = Tensor(np.random.normal(loc=0.0, scale=std_normal, size=weight_shape), mindspore.float32)
        m.weight.set_data(weights)

        out_im = m(input_im)

        observed = out_im[0]
        observed = ops.Transpose()(observed, (1, 2, 0)) 
        observed = observed.asnumpy()

        observed = alpha * image_float + (1 - alpha) * observed
        observed = np.clip(observed, 0.0, 1.0)
        print('max, min = ', np.amax(observed), np.amin(observed))

        im1.set_data(observed)
        plt.pause(1.0)

        if save_images:
            plt.savefig("rand_conv_ms_{:04d}.jpg".format(i))

    plt.show()

if __name__ == "__main__":
    fire.Fire(do_rand_conv)
