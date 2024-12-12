import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, Tensor
from mindspore.dataset.vision import py_transforms as transforms
import numpy as np
import cv2
from unet import UNet


if __name__ == '__main__':
    img_dir = r'./test.png'
    model_dir = r'./best_ms.ckpt'
    trans = transforms.ToTensor()
    device = ms.context.get_context("device_target")
    model = UNet()
    checkpoint = load_checkpoint(model_dir)
    load_param_into_net(model, checkpoint)
    model.set_train(False)

    input_img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    input_tensor = trans(input_img)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = Tensor(input_tensor, ms.float32)

    output = model(input_tensor)
    print(output[0][0])
    output = ms.ops.clip_by_value(output, 0.0, 1.0)

    output = output.asnumpy()[0, 0, :, :]
    output = np.array(output * 255, dtype=np.uint8)
    cv2.imwrite('./output_ms.png', output)
