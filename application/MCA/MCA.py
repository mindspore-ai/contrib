import numpy as np
from mindspore import nn, Tensor
import mindspore.ops as ops

class MomentEfficient(nn.Cell):
    def construct(self, x):
        avg_x = ops.mean(x, (2, 3)).unsqueeze(-1).permute(0,2,1)  # bs, c, 1, 1
        std_x = ops.std(x, (2, 3), ddof=0).unsqueeze(-1).permute(0,2,1)   # bs, c, 1, 1
        moment_x = ops.cat((avg_x, std_x), axis=1) # bs, 2, c, 1
        return moment_x # bs, c, 2

class MomentStrong(nn.Cell):
    def construct(self, x):
        n = x.shape[2] * x.shape[3]
        avg_x = ops.mean(x, (2, 3), keep_dims=True)  # bs, c, 1, 1
        std_x = ops.std(x, (2,3), ddof=0, keepdims=True)  # bs, c, 1, 1
        skew_x1 = (x - avg_x) ** 3 / n
        skew_x2 = std_x ** 3
        skew_x = ops.sum(skew_x1,(2,3), keepdim=True)/(skew_x2 + 1e-5) # bs,c,1,1
        avg_x = avg_x.squeeze(-1).permute(0,2,1)
        skew_x = skew_x.squeeze(-1).permute(0,2,1)        
        moment_x = ops.cat((avg_x, skew_x), axis=1) # bs,*,c
        return moment_x 


class ChannelAttention(nn.Cell):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        k = 3
        self.conv = nn.Conv1d(2, 1, kernel_size=k, stride=1, padding=(k-1)//2, pad_mode='pad')  
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        y = self.conv(x)
        output = self.sigmoid(y)
        return output

class MomentAttentionV1(nn.Cell):
    def __init__(self):
        super(MomentAttentionV1, self).__init__()
        self.moment = MomentEfficient()
        self.c = ChannelAttention()

    def construct(self, x):
        y = self.moment(x)  # bs, 2, c
        result = self.c(y)  # bs, 1, c
        result = result.permute(0, 2, 1).unsqueeze(-1)  # bs, c, 1, 1
        return x * result.expand_as(x)

class MomentAttentionV2(nn.Cell):
    def __init__(self):
        super(MomentAttentionV2, self).__init__()
        self.moment = MomentStrong()
        self.c = ChannelAttention()

    def construct(self, x):
        y = self.moment(x)  # bs, 2, c
        result = self.c(y)  # bs, 1, c
        result = result.permute(0, 2, 1).unsqueeze(-1)  # bs, c, 1, 1
        return x * result.expand_as(x)

if __name__ == '__main__':
    input_tensor = Tensor(np.random.randn(50, 512, 7, 7).astype(np.float32))
    mca = MomentAttentionV1()
    output = mca(input_tensor)
    print(output.shape)  

    mca2 = MomentAttentionV2()
    output2 = mca2(input_tensor)
    print(output2.shape)  
