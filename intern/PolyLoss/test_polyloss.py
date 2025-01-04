import unittest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from polyloss_mindspore import PolyBCELoss


class TestPolyBCELoss(unittest.TestCase):
    def test1(self):
        input1 = Tensor(
            [[[0.6970, 0.6716, 0.1472],
              [0.7298, 0.6212, 0.0785],
              [0.4674, 0.2362, 0.0147]],
             [[0.4235, 0.4401, 0.1257],
              [0.1033, 0.3107, 0.2389],
              [0.2773, 0.3202, 0.9246]]],
            ms.float32
        )
        target1 = Tensor(
            [[[0., 0., 0.],
              [1., 1., 1.],
              [1., 0., 0.]],
             [[1., 0., 0.],
              [1., 0., 0.],
              [1., 1., 1.]]],
            ms.float32
        )
        polybceloss = PolyBCELoss()
        result = polybceloss(input1, target1)
        np.testing.assert_allclose(
            result.asnumpy(), 1.1753, atol=1e-4, rtol=1e-4
        )

    def test2(self):
        input1 = Tensor(
            [[[0.6970, 0.6716, 0.1472],
              [0.7298, 0.6212, 0.0785],
              [0.4674, 0.2362, 0.0147]],
             [[0.4235, 0.4401, 0.1257],
              [0.1033, 0.3107, 0.2389],
              [0.2773, 0.3202, 0.9246]]],
            ms.float32
        )
        target1 = Tensor(
            [[[0., 0., 0.],
              [1., 1., 1.],
              [1., 0., 0.]],
             [[1., 0., 0.],
              [1., 0., 0.],
              [1., 1., 1.]]],
            ms.float32
        )
        polybceloss = PolyBCELoss(epsilon=0)
        bceloss = ms.nn.BCEWithLogitsLoss()
        result = polybceloss(input1, target1)
        expected = bceloss(input1, target1)
        np.testing.assert_allclose(
            result.asnumpy(), expected.asnumpy(), atol=1e-4, rtol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
