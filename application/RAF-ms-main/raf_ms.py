import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import ops


class RaF(nn.Cell):
    """
    Implementation of the RAF activation function. See: https://arxiv.org/pdf/2208.14111.pdf
    """
    def __init__(self, m, n):
        super().__init__()
        self.m = m
        self.n = n

        # 创建并初始化 m_weights
        self.m_weights = ms.Parameter(ms.Tensor(shape=(self.m,), dtype=ms.float32, init=Normal()))
        # 创建并初始化 n_weights
        self.n_weights = ms.Parameter(ms.Tensor(shape=(self.n,), dtype=ms.float32, init=Normal()))


    def construct(self, inputs):
        for i in range(self.m):
            if i == 0:
                x = ops.pow(inputs, ms.Tensor([i])) * self.m_weights[i]
            else:
                x += ops.pow(inputs, ms.Tensor([i])) * self.m_weights[i]

        for i in range(self.n):
            if i == 0:
                x2 = ops.pow(inputs, ms.Tensor([i])) * self.n_weights[i]
            else:
                x2 += ops.pow(inputs, ms.Tensor([i])) * self.n_weights[i]

        x2 = 1 + ops.abs(x2)
        return x/x2

import unittest
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore import ops
class TestRaF(unittest.TestCase):
    def test_raf_init(self):
        """测试RaF类的初始化"""
        m, n = 3, 4
        raf = RaF(m, n)
        self.assertEqual(raf.m, m)
        self.assertEqual(raf.n, n)
        self.assertEqual(raf.m_weights.shape, (m,))
        self.assertEqual(raf.n_weights.shape, (n,))

    def test_raf_construct(self):
        """测试RaF类的construct方法"""
        m, n = 2, 3
        raf = RaF(m, n)
        input_tensor = ms.Tensor([2.0, 3.0], dtype=ms.float32)

        # 此处使用随机初始化，所以对于相同的输入，每次的输出可能不一样，
        # 只能测试其结构和预期不一致的行为，如无异常抛出等
        output = raf.construct(input_tensor)

        # 检查输出是否是Tensor
        self.assertIsInstance(output, ms.Tensor)

        # 这里可以添加更多的断言来检查输出值是否符合预期，
        # 但因为权重是随机初始化的，且涉及幂运算和绝对值，所以精确值比较复杂

    # 可以继续添加更多的测试用例来覆盖更多边缘情况

if __name__ == '__main__':
    unittest.main()
