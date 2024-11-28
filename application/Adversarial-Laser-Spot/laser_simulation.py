import cv2
import numpy as np
import random
from PIL import Image
import mindspore as ms
from mindspore import Tensor, context
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindcv.models import resnet50

context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if ms.get_context('device_target') == 'GPU' else 'CPU')

net = resnet50(pretrained=True, num_classes=1000)
net.set_train(False)

def classify(dir, net):
    """
    分类函数：使用 MindCV 的 resnet50 模型对输入图像进行分类。
    """
    img = Image.open(dir).convert("RGB")
    img = np.array(img)

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    transform = transforms.Compose([
        vision.Resize(256),                    # 调整大小
        vision.CenterCrop(224),               # 裁剪为中心区域
        vision.Normalize(mean=mean, std=std), # 标准化
        vision.HWC2CHW()                      # HWC -> CHW 格式
    ])
    img = transform(img)
    img = np.expand_dims(img, axis=0)  # 增加 batch 维度

    img = Tensor(img, ms.float32)

    output = net(img)
    output = output.asnumpy().flatten()

    I = np.argsort(output)[::-1][:10]
    label = I[0]

    return label

def img_neon_effect(img, width, height, filenameSize):
    """
    图像霓虹效果函数
    """
    for i in range(28):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        cv2.ellipse(img, (x, y), (1, 1), 0, 0, 360, (b, 255, r), 3, 4)
        cv2.imwrite("54_" + filenameSize + ".jpg", img)
        img = cv2.imread("54_" + filenameSize + ".jpg")
        cv2.circle(img, (x, y), 0, (255, 255, 255), -1)
        cv2.imwrite("result.jpg", img)

def img_laser_effect(img, r, g, b, width, height, path):
    """
    图像激光效果函数
    """
    for i in range(35):
        x = random.randint(0, width)
        y = random.randint(0, height)

        cv2.ellipse(img, (x, y), (1, 1), 0, 0, 360, (r, g, b), 3, 4)
        cv2.imwrite(path, img)
        img = cv2.imread(path)
        cv2.circle(img, (x, y), 0, (255, 255, 255), -1)
        cv2.imwrite(path, img)

if __name__ == "__main__":
    img_path = "green_lizard.jpg"
    label = classify(img_path, net)
    print(f"预测的类别标签为: {label}")

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    img_neon_effect(img, width, height, "neon_output")
    img_laser_effect(img, 255, 0, 0, width, height, "laser_output.jpg")
