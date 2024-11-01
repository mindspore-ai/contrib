
from einops.layers.torch import Rearrange, Reduce
import mindspore as ms
from mindspore import nn,ops
# helpers

def pair(val):
    if not isinstance(val, tuple):
        raise TypeError("Input must be a tuple")
    return val

# classes

class Affine(nn.Cell):
    def __init__(self, axis):
        super().__init__()
        self.g = ms.Parameter(ops.ones(1, 1, axis))
        self.b = ms.Parameter(ops.zeros(1, 1, axis))

    def construct(self, x):
        return x * self.g + self.b

class PreAffinePostLayerScale(nn.Cell): # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = ops.zeros(1).fill_(init_eps)
        self.scale = ms.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

def ResMLP(*, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4):
    image_height, image_width = pair(image_size)
    assert (image_height % patch_size) == 0 and (image_width % patch_size) == 0, 'image height and width must be divisible by patch size'
    num_patches = (image_height // patch_size) * (image_width // patch_size)
    wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

    return nn.SequentialCell(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Dense((patch_size ** 2) * 3, dim),
        *[nn.SequentialCell(
            wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
            wrapper(i, nn.SequentialCell(
                nn.Dense(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Dense(dim * expansion_factor, dim)
            ))
        ) for i in range(depth)],
        Affine(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Dense(dim, num_classes)
    )




import unittest


class TestPairFunction(unittest.TestCase):

    def test_pair_with_tuple(self):
        self.assertEqual(pair((3, 4)), (3, 4))

    def test_pair_with_list(self):
        with self.assertRaises(TypeError):
            pair([1, 2])

    def test_pair_with_string(self):
        with self.assertRaises(TypeError):
            pair("test")




if __name__ == '__main__':
    unittest.main()
