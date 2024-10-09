import mindspore.ops as ops
import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
class RiemannNoise(nn.Cell):
    def __init__(self, size:int, channels:int):
        super(RiemannNoise, self).__init__()
        '''
        Initializes the module, taking 'size' as input for defining the matrix param.
        '''
        device_target="CPU"
        context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)
        self.device=device_target
        if type(size) == int:
            h = w = size
        elif type(size) == tuple and (type(x)==int for x in size):
            h = size[-2]
            w = size[-1]
        else:
            raise ValueError("Module must be initialized with a valid int or tuple of ints indicating the input dimensions.")
        self.param1=ms.Parameter(ops.normal((h, w), 0, 1).astype(ms.float32)),
        self.param2=ms.Parameter(ms.numpy.zeros(channels).astype(ms.float32)),
        self.param3=ms.Parameter(ops.normal((h, w), 0, 1).astype(ms.float32)),
        self.param4=ms.Parameter(ms.numpy.full((1,), 0.5).astype(ms.float32)),
        self.param5=ms.Parameter(ms.numpy.full((1,), 0.5).astype(ms.float32)),
        self.param6=ms.Parameter(ms.numpy.full((1,), 0.5).astype(ms.float32)),
        self.params = [
            self.param1,
            self.param2,
            self.param3,
            self.param4,
            self.param5,
            self.param6
        ]

        self.noise = ops.zeros(1)

    def construct(self, x):
        N, c, h, w = x.shape
        A, ch, b, alpha, r, w = self.params
        s, _ = ops.max(-x, axis=1, keepdims=True)
        s = s - s.mean(axis=(2, 3), keep_dims=True)
        s_max = ops.abs(s).amax(axis=(2, 3), keepdims=True)
        s = s / (s_max + 1e-8)
        s = (s + 1) / 2
        s = s * A + b
        s = ops.tile(s, (1, c, 1, 1))

        sp_att_mask = alpha[0] + (1 - alpha[0]) * s
        sp_att_mask = sp_att_mask * ops.rsqrt(
            ops.mean(ops.square(sp_att_mask), axis=(2, 3), keep_dims=True) + 1e-8)
        sp_att_mask = r * sp_att_mask
        # 添加噪声
        noise = ops.StandardNormal()(x.shape)
        x = x + (noise * w)
        x = x * sp_att_mask
        return x

# test_riemann_noise.py
import unittest
import mindspore as ms
from mindspore import Tensor


class TestRiemannNoise(unittest.TestCase):

    def setUp(self):
        self.size = 10
        self.channels = 3
        self.rn = RiemannNoise(self.size, self.channels)
        self.x = Tensor(ms.ops.randn(1, self.channels, self.size, self.size), dtype=ms.float32)

    def test_riemann_noise_init(self):
        # 测试正常情况
        self.assertIsInstance(self.rn, RiemannNoise)
        self.assertEqual(len(self.rn.params), 6)

        # 测试错误情况（size不是整数也不是整数元组）
        with self.assertRaises(ValueError):
            RiemannNoise("not an int or tuple", self.channels)


    def test_riemann_noise_forward(self):
        # 测试forward方法
        y = self.rn(self.x)
        self.assertEqual(y.shape, self.x.shape)

if __name__ == '__main__':
    unittest.main()
