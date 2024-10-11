import argparse

import mindspore as ms
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
from PIL import Image
from mindspore import ops

from swin_transformer.model import swin_base_patch4_window7_224


class SwinTransformer(nn.Cell):
    def __init__(self, num_features=512):
        super(SwinTransformer, self).__init__()
        self.backbone = swin_base_patch4_window7_224()
        self.num_features = num_features
        self.feat = nn.Dense(1024, num_features) if num_features > 0 else None

    def construct(self, x):
        x = self.backbone.construct_features(x)
        if self.feat is not None:
            x = self.feat(x)
        return x

    def load_model(self, model_path):
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(self, param_dict)


class DataProcessor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transformer = transforms.Compose([
            vision.Resize((self.height, self.width)),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False),
        ])

    def __call__(self, img):
        return ms.Tensor(self.transformer(img)[0]).unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISR')
    parser.add_argument('--model-weight', type=str, help='the path of model weight')
    parser.add_argument('--image1', type=str, default='./image1.jpg', help='the path of image 1')
    parser.add_argument('--image2', type=str, default='./image2.jpg', help='the path of image 2')
    args = parser.parse_args()

    data_processor = DataProcessor(height=224, width=224)
    model = SwinTransformer(num_features=512)
    model.set_train(False)

    if args.model_weight:
        model.load_model(args.model_weight)

    image1 = args.image1
    image2 = args.image2

    image1 = data_processor(Image.open(image1).convert('RGB'))
    image2 = data_processor(Image.open(image2).convert('RGB'))

    l2_normalize = ops.L2Normalize(axis=1)
    A_feat = l2_normalize(model(image1))
    B_feat = l2_normalize(model(image2))
    similarity = A_feat.matmul(B_feat.transpose(1, 0))
    print("The similarity is {}".format(similarity[0, 0]))
