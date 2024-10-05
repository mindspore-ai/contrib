import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

class GeometryPrior(nn.Cell):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * ops.rand(1, 2, k, k) - 1
        self.l1 = nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = nn.Conv2d(int(multiplier * channels), channels, 1)
        
    def construct(self, x):
        x = self.l2(ops.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, self.k ** 2)

class KeyQueryMap(nn.Cell):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = nn.Conv2d(channels, channels // m, 1)
    
    def construct(self, x):
        return self.l(x)

class AppearanceComposability(nn.Cell):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.padding = padding
        self.stride = stride
    
    def construct(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = ops.unfold(key_map, k, 1, self.padding, self.stride)
        query_map_unfold = ops.unfold(query_map, k, 1, self.padding, self.stride)
        key_map_unfold = key_map_unfold.view(
                    key_map.shape[0], key_map.shape[1],
                    -1,
                    key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
                    query_map.shape[0], query_map.shape[1],
                    -1,
                    query_map_unfold.shape[-2] // query_map.shape[1])
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]


def combine_prior(appearance_kernel, geometry_kernel):
    return ops.softmax(appearance_kernel + geometry_kernel,
                                       axis=-1)

class LocalRelationalLayer(nn.Cell):
    def __init__(self, channels, k, stride=1, m=None, padding=0):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.kmap = KeyQueryMap(channels, k)
        self.qmap = KeyQueryMap(channels, k)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels//self.m)
        self.final1x1 = nn.Conv2d(channels, channels, 1)
        
    def construct(self, x):
        gpk = self.gp(0)
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))
        gpk = gpk.view(ak.shape)
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        print(f"ck shape before indexing: {ck.shape}")
        x_unfold = ops.unfold(x, k, 1, self.padding, self.stride)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // m,
                                 -1, x_unfold.shape[-2] // x.shape[1])
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1],
                                          -1, x_unfold.shape[-2] // x.shape[1])
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                                                            self.stride + 1                               
        pre_output = ops.sum(pre_output, axis=-1).view(x.shape[0], x.shape[1],
                                                         h_out, w_out)
        return self.final1x1(pre_output)
    
def main():
    channels = 64
    k = 3
    stride = 1
    m = 8
    padding = 1
    model = LocalRelationalLayer(channels, k, stride, m, padding)
    print(model)

if __name__ == "__main__":
    main()