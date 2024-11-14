import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, HeNormal, Constant
import numpy as np

def conv(ni, nf, s=2, ks=3, act=nn.ReLU(), bn=True, bn_before_act=False, **kwargs):
    '''这个函数将卷积层、激活函数和可选的BatchNorm层封装在一个Sequential模块中'''
    layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=s, pad_mode='pad', padding=ks//2, has_bias=False)]
    if act:
        layers.append(act)
    if bn:
        layers.append(nn.BatchNorm2d(nf, **kwargs))
    if act and bn and bn_before_act:
        layers[-2], layers[-1] = layers[-1], layers[-2]
    return nn.SequentialCell(layers)

def noop(x):
    return x

class Flatten(nn.Cell):
    '''将输入展平，例如，如果输入形状为(32,64,4,4)，输出形状为(32,1024)'''
    def construct(self, x):
        return x.view(x.shape[0], -1)

def init_cnn(func, m):
    '''使用指定的初始化函数初始化模块m'''
    if isinstance(m, (nn.Dense, nn.Conv2d)):
        func(m.weight)
    else:
        for cell in m.cells():
            init_cnn(func, cell)

def fixup_init(a, m=2, num_layers=1, mode='fan_in', zero_wt=False):
    '''使用Fixup初始化初始化层a'''
    if isinstance(a, nn.Conv2d):
        w = a.weight
        if zero_wt:
            w.set_data(initializer(Constant(0), w.shape, w.dtype))
        else:
            s = w.shape
            c1 = s[1] if mode == 'fan_in' else s[0]
            c = s[2] * s[3]
            std = math.sqrt(2 / (c * c1)) * num_layers ** (-0.5 / (m - 1))
            w.set_data(initializer(HeNormal(negative_slope=0, mode=mode), w.shape, w.dtype) * std)

class Resblock1(nn.Cell):
    '''根据He等人的论文"Bag of Tricks for Image Classification with Convolutional Neural Networks"构建下采样块或恒等块'''
    def __init__(self, ni, nf, stride, activation=nn.ReLU(), expansion=4, init='Fixup', num_layers=1):
        super(Resblock1, self).__init__()
        self.init_type = init
        bn = False if init == 'Fixup' else True
        nconvs = 2 if expansion == 1 else 3
        if nconvs > 2:
            nh = nf
            ni, nf = expansion * ni, expansion * nf
        else:
            nh = nf
        nfs = [ni] + [nf] * (nconvs) if nconvs <= 2 else [nh] * (nconvs - 1) + [nf]
        layers = [conv(ni, nh, s=1, ks=1, bn=bn)] if nconvs > 2 else []
        layers += [conv(nfs[i], nfs[i+1],
                        s=stride if i == 0 else 1,
                        act=None if i == len(nfs) - 2 else activation,
                        ks=1 if (i != 0 and nconvs > 2) else 3,
                        bn=bn)
                   for i in range(len(nfs) - 1)]
        self.activation = activation
        self.convs = nn.SequentialCell(layers)
        self.sc = noop if ni == nf else conv(ni, nf, s=1, ks=1, act=None, bn=bn)
        self.pool = noop if stride == 1 else nn.AvgPool2d(kernel_size=2, stride=2)
        if init == 'Fixup':
            for i in range(len(self.convs)):
                fixup_init(self.convs[i][0], m=nconvs, num_layers=num_layers,
                           zero_wt=True if i == len(self.convs) - 1 else False)

            for i in range(nconvs * 2):
                setattr(self, f'bias_{i}', Parameter(Tensor(np.zeros(1), mindspore.float32)))
            self.scale = Parameter(Tensor(np.ones(1), mindspore.float32))

    def construct(self, x):
        if self.init_type != 'Fixup':
            return self.activation(self.convs(x) + self.sc(self.pool(x)))
        sc = self.sc(self.pool(x + getattr(self, 'bias_0')))
        for i in range(0, len(self.convs) * 2, 2):
            k = i // 2
            x = self.convs[k][0](x + getattr(self, f'bias_{i}'))
            if len(self.convs[k].cells()) > 1:
                x = self.convs[k][1](x + getattr(self, f'bias_{i + 1}'))
        out = self.scale * x + getattr(self, f'bias_{i + 1}')
        return self.activation(out + sc)

class ResnetModule(nn.Cell):
    '''构建整个ResNet模型。cin=输入的通道数，cout=分类的类别数，init_func=您想要使用的初始化函数'''
    def __init__(self, expansions, layers, cin=3, cout=10, init_func='kaiming_normal_', **kw):
        super(ResnetModule, self).__init__()
        nfs = [cin, 32, 32, 64]
        self.ifunc = init_func
        in_block = [conv(nfs[i], nfs[i + 1], s=2 if i == 0 else 1,
                         bn=False if self.ifunc == 'Fixup' else True) for i in range(len(nfs) - 1)]
        l = [64 // expansions, 64, 128, 256, 512]
        blocks = [self._make_blocks(l[i], l[i + 1], s=1 if i == 0 else 2, num=n, init=self.ifunc, expansions=expansions,
                                    num_layers=sum(layers)) for i, n in enumerate(layers)]
        self.lin = nn.Dense(l[-1] * expansions, cout)

        self.model = nn.SequentialCell(
            *in_block,
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'),
            *blocks,
            nn.AdaptiveAvgPool2d(output_size=1),
            Flatten(),
            self.lin
        )

        if self.ifunc == 'Fixup':
            self.lin.weight.set_data(initializer(Constant(0), self.lin.weight.shape, self.lin.weight.dtype))
            self.lin.bias.set_data(initializer(Constant(1), self.lin.bias.shape, self.lin.bias.dtype))
            for i in range(len(in_block)):
                setattr(self.model, f'rbias_{i}', Parameter(Tensor(np.zeros(1), mindspore.float32)))
            setattr(self.model, 'linear_bias', Parameter(Tensor(np.zeros(1), mindspore.float32)))
        else:
            func = getattr(mindspore.common.initializer, self.ifunc)
            init_cnn(func, self.model)

    @staticmethod
    def _make_blocks(ni, nf, s, num, init, num_layers, expansions=1):
        l = [ni] + [nf] * num
        return nn.SequentialCell(*[Resblock1(l[i], l[i + 1], stride=s if i == 0 else 1,
                                             expansion=expansions, num_layers=num_layers, init=init)
                                   for i in range(len(l) - 1)])

    def construct(self, x):
        if self.ifunc != 'Fixup':
            return self.model(x)
        i = 0
        while not isinstance(self.model[i], nn.MaxPool2d):
            x = self.model[i][1](self.model[i][0](x) + getattr(self.model, f'rbias_{i}'))
            i += 1

        for j in range(i, len(self.model)):
            if isinstance(self.model[j], nn.Dense):
                x = self.model[j](x + getattr(self.model, 'linear_bias'))
            else:
                x = self.model[j](x)
        return x

def resnet18(**kwargs):
    return ResnetModule(1, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResnetModule(1, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResnetModule(4, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResnetModule(4, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResnetModule(4, [3, 8, 36, 3], **kwargs)

if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    input_tensor = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)
    for model_name, model_func in [('resnet18', resnet18),
                                   ('resnet34', resnet34),
                                   ('resnet50', resnet50),
                                   ('resnet101', resnet101),
                                   ('resnet152', resnet152)]:
        net = model_func(cin=3, cout=10, init_func='HeNormal')
        output = net(input_tensor)
        print(f"{model_name} output shape: {output.shape}")