import mindspore
from mindspore import nn, ops

class GroupBatchnorm2d(nn.Cell):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super().__init__()
        self.group_num = group_num
        self.gamma = mindspore.Parameter(ops.ones((c_num, 1, 1)))
        self.beta = mindspore.Parameter(ops.zeros((c_num, 1, 1)))
        self.eps = eps

    def construct(self, x:mindspore.Tensor):
        N, C, H, W = x.shape

        x = x.view(N, self.group_num, -1)
        
        mean = x.mean(axis = 2, keep_dims = True)
        std = x.std(axis = 2, keepdims = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta
    

if __name__ == "__main__":
    cnum = 4
    batch_size = 16
    model = GroupBatchnorm2d(cnum)
    Input = ops.rand([batch_size, cnum, 16, 16])
    print(model(Input))