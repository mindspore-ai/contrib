import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore.dataset import transforms
import mindspore.dataset.vision as tools
from mindspore import context
from PIL import Image
import numpy as np
import os
import random
import kagglehub
import tqdm



class ReNet(nn.Cell):
    def __init__(self, input_channels, hidden_size, patch_size):
        super(ReNet, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.patch_area = patch_size * patch_size
        self.patch_dim = input_channels * self.patch_area
        self.vertical_rnns_forward = nn.LSTM(self.patch_dim,hidden_size,batch_first=True)
        self.vertical_rnns_backward = nn.LSTM(self.patch_dim,hidden_size,batch_first=True)
        self.horizontal_rnns_forward = nn.LSTM(hidden_size*2,hidden_size,batch_first=True)
        self.horizontal_rnns_backward = nn.LSTM(hidden_size*2,hidden_size,batch_first=True)
    # 水平
    def vertical_sweep(self, patches):
        batch_size, c,L = patches.shape
        patches = ops.transpose(patches,(0,2,1))
        # bacth,L,c
        # 随机生成h0和c0
        h0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        c0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        output_f ,_= self.vertical_rnns_forward(patches,(h0,c0)) #(batch,L,hidden_size)
        # (b,L,c)
        # 反转重新扫描
        reversed_patches = ops.reverse(patches, axis=(1,))
        h0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        c0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        output_b ,_= self.vertical_rnns_backward(reversed_patches,(h0,c0)) #(batch,L,hidden_size)
        # print("in renet1")
        # print(output_b.shape)
        output = ops.Concat(axis=2)((output_f, output_b))
        # (b,L,2c)
        return output

    def horizontal_sweep(self, vertical_output):

        batch_size,c,L = vertical_output.shape
        patches = ops.transpose(vertical_output,(0,2,1))
        # (b,L,c)

        h0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        c0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        output_f ,_= self.horizontal_rnns_forward(patches,(h0,c0)) #(batch,L,hidden_size)
        # (b,L,c)
        # 反转重新扫描
        reversed_patches = ops.reverse(patches, axis=(1,))
        h0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        c0 = mindspore.Tensor(np.random.randn(1,batch_size, self.hidden_size), mindspore.float32)
        output_b ,_= self.horizontal_rnns_backward(reversed_patches,(h0,c0)) #(batch,L,hidden_size)

        output = ops.Concat(axis=2)((output_b, output_f))
        # (b,L,2c)
        return output

    def construct(self, x):
        batch_size, channels, height, width = x.shape
        # 滑动窗口
        x = ops.unfold(x,kernel_size=self.patch_size, stride=self.patch_size)
        # (batch,c,L)
        vertical_output = self.vertical_sweep(x)
        
        # 转回图
        vertical_output = ops.transpose(vertical_output,(0,2,1))
        # (batch,L,2hidden)
        # 转回二维 (batch,h,w,2hidden)
        size = mindspore.Tensor((height//self.patch_size,width//self.patch_size),dtype=mindspore.int32)
        patches = ops.fold(vertical_output,output_size=size,kernel_size=1,stride=1)
        
        # 转置
        patches = ops.transpose(patches,(0,1,3,2))
        # 滑动窗口
        
        x = ops.unfold(patches,kernel_size=1, stride=1)
        
        # (b,2hidden,L)
        horizontal_output = self.horizontal_sweep(x)
        # (b,L,2c)
        patches = ops.transpose(horizontal_output,(0,2,1))
        
        patches = ops.fold(patches,output_size=size,kernel_size=1,stride=1)
        
        return patches



# 定义基于VGG的预处理层
class VGGPreprocessing(nn.Cell):
    def __init__(self):
        super(VGGPreprocessing, self).__init__()
        # 定义均值张量，值为VGG网络常用的图像通道均值，不需要梯度更新
        self.mean = mindspore.Parameter(mindspore.Tensor([0.485, 0.456, 0.406]), requires_grad=False)
        # 定义标准偏差张量，值为VGG网络常用的图像通道标准偏差，不需要梯度更新
        self.std = mindspore.Parameter(mindspore.Tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def construct(self, x):

        mean_adjusted = self.mean.view(1, -1, 1, 1).expand_as(x)
        std_adjusted = self.std.view(1, -1, 1, 1).expand_as(x)
        # 根据调整后的均值和标准偏差对x进行标准化处理
        x = (x - mean_adjusted) / std_adjusted
        return x


# ReSeg模型
class ReSeg(nn.Cell):
    def __init__(self):
        super(ReSeg, self).__init__()
        self.vgg = VGGPreprocessing()
        # rnn层用于特征提取
        self.renet1 = ReNet(3,256,2)
        self.renet2 = ReNet(512,256,2)
        self.renet3 = ReNet(512,256,2)
        # 反卷积层用于上采样恢复图像大小并得到分割结果
        self.deconv1 = nn.Conv2dTranspose(512, 32, kernel_size=32,stride = 2)
        self.deconv2 = nn.Conv2dTranspose(32, 1, kernel_size=32, stride = 4)

        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()

    def construct(self, x):
        # 特征提取
        # print(x.shape)
        x = self.vgg(x)
        x = self.renet1(x)
        x = self.renet2(x)
        x = self.renet3(x)
        # 上采样得到分割结果
        
        
        x = self.relu(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)

        x = self.sigmod(x)

        return x
    

# test
# model = ReSeg()
# images = mindspore.Tensor(np.random.randn(1,3,32,32),dtype=mindspore.float32)
# print(model.construct(images).shape)