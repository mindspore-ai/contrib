"""Train style transfer"""

from typing import Type, Union, Tuple
import mindspore as ms
from mindspore import nn, ops, context
import mindspore.dataset.transforms.py_transforms as transforms
import mindspore.dataset.vision.py_transforms as py_vision
from mindvision.classification.models import resnet18

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_image(img_path, max_size=400, shape=None):
    """
    load image and preprocessing
    """

    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    trans = transforms.Compose([
        py_vision.Resize(size),
        # py_vision.CenterCrop(224),
        py_vision.ToTensor(output_type=np.float32), #HWC->CHW [0, 255]->[0.0, 1.0]
        # py_vision.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        py_vision.Normalize(mean=mean, std=std)   # Or using this mean and std
    ])
    image = ms.Tensor(trans(image))[:3, :, :]
    return image

class Net(nn.Cell):
    """
    MobileNet V1 backbone.
    """

    def __init__(self):
        super(Net, self).__init__()
        backbone = resnet18(pretrained=True).backbone
        layers = list(backbone.cells())
        self.features = nn.CellList(layers)

    def construct(self, x):
        """Forward pass"""

        features = ()
        for block in self.features:
            x = block(x)
            features = features + (x,)

        return features


# 正则化除以feature的尺寸
def size_mormlized(weights, tensors):
    '''
    :param: style_weights: list
    :param: tensors: tuple of tensors
    '''

    assert len(tensors) == len(weights)

    for i in range(len(tensors)):
        _, d, h, w = tensors[i].shape
        size = 4 * d * d * h * h * w * w
        weights[i] /= size

    return weights


class TargetNet(nn.Cell):
    """
    construct to train new target image
    """

    def __init__(self, cont, frozen_backbone):
        super(TargetNet, self).__init__()
        self.parameter = ms.Parameter(cont.copy(), name='w', requires_grad=True)
        self.backbone = frozen_backbone

    def construct(self):
        return self.backbone(self.parameter)


class MSELossWithCell(nn.LossBase):
    """
    define loss net
    """

    def __init__(self, net):
        super(MSELossWithCell, self).__init__()
        self.square1 = ops.Square()
        self.square2 = ops.Square()
        self.f_x = net # backbone
        self.content_weight = 1
        self.style_weight = 1e9

    def construct(self, content_f, style_f):
        """
        forward compute loss
        """

        target_features = self.f_x() # tuple
        content_loss = self.square1(target_features[3] - content_f[3]) # fetch index of 3rd
        content_loss = self.get_loss(content_loss) # reduce_mean

        target_grams = list(map(self.gram_matrix, target_features)) # gram_matrix(target_features)# list of 6 matrics.
        style_grams = list(map(self.gram_matrix, style_f)) # list of tensors

        total_style_loss = 0
        for i in range(len(target_grams)):
            style_loss = self.square2(target_grams[i] - style_grams[i])
            style_loss = self.get_loss(style_loss, weights=style_weights[i])
            total_style_loss += style_loss

        total_loss = self.content_weight * content_loss + self.style_weight * total_style_loss
        # print(f'total_style_loss:{total_style_loss}')
        # print(f'content:{content_loss}')

        return total_loss

    def backbone_network(self):
        return self.f_x


    def gram_matrix(self, tensor: Type[Union[Tuple[ms.Tensor], ms.Tensor]]):
        '''
        gramm matrix
        :param: tensor: tensor or tuple of tensor
        '''

        _, d, h, w = tensor.shape

        reshape = ops.Reshape()
        tensor = reshape(tensor, (d, h*w))

        matmul = ops.MatMul(transpose_a=False, transpose_b=True)
        gram = matmul(tensor, tensor)

        return gram


class MyTrainStep(nn.TrainOneStepCell):
    """
    construct train step net
    """

    def __init__(self, network, optimizer):
        """
        network: backbone with loss
        optimizer: instance of nn.Adam
        """

        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, x, y):
        weights = self.weights #self.weights = self.optimizer.parameters
        cost = self.network(x, y)
        grads = self.grad(self.network, weights)(x, y)

        return cost, self.optimizer(grads)

def im_convert(tensor):
    """
    convert ensor to Image
    """

    image = tensor.asnumpy().squeeze() #mindspore
    image = image.transpose(1, 2, 0)
    image = image * np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    return image


if __name__ == '__main__':

    # 训练设置
    show_every = 500
    steps = 5000

    # 加载训练图片
    content = load_image('F2.jpg')
    style = load_image('xingkong.jpeg', shape=content.shape[-2:])
    print(content.shape)
    print(style.shape)

    #定义backbone对象
    frozeon_net = Net()
    for param in frozeon_net.get_parameters():
        param.requires_grad = False

    target_net = TargetNet(content, frozeon_net)

    # 提取backbone中间层特征
    content_features = frozeon_net(content)
    style_features = frozeon_net(style)

    style_weights = [1.0, 0.75, 0.75, 0.2, 0.2, 0.2]
    style_weights = size_mormlized(style_weights, style_features)
    content_weight = 1
    style_weight = 1e9

    #定义优化器net.trainable_params()
    opt = nn.Adam(params=target_net.trainable_params(), learning_rate=0.003)

    net_with_criterion = MSELossWithCell(target_net)  # 构建损失网络 前向传播
    train_net = MyTrainStep(net_with_criterion, opt)     # 构建训练网络 反向传播

    for step in range(1, steps+1):
        train_net(content_features, style_features)                  # 执行训练，并更新权重
        loss = net_with_criterion(content_features, style_features)  # 计算损失值
        if step % show_every == 0:
            print("Total loss: ", loss)
            # plt.imshow(im_convert(ms.Tensor(target_net.parameter)))
            # plt.show()


    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, figsize=(15, 15))
    ax1.imshow(im_convert(content))
    ax1.set_title("Content Image", fontsize=20)
    ax3.imshow(im_convert(style))
    ax3.set_title("Style Image", fontsize=20)
    ax2.imshow(im_convert(ms.Tensor(target_net.parameter)))
    ax2.set_title("Stylized Target Image", fontsize=20)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    # plt.show()
    plt.savefig('target.jpg')
