import numpy as np
import mindspore as ms
from mindspore import ops,nn

class Fastformer(nn.Cell):
    def __init__(self, dim = 3, decode_dim = 16):
        super(Fastformer, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Dense(dim, decode_dim * 3, has_bias = False)
        self.weight_q = nn.Dense(dim, decode_dim, has_bias = False)
        self.weight_k = nn.Dense(dim, decode_dim, has_bias = False)
        self.weight_v = nn.Dense(dim, decode_dim, has_bias = False)
        self.weight_r = nn.Dense(decode_dim, decode_dim, has_bias = False)
        self.weight_alpha = ms.Parameter(ops.randn(decode_dim))
        self.weight_beta = ms.Parameter(ops.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def construct(self, x, mask = None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape
        
        mask_value = -ms.tensor(np.finfo(ms.dtype_to_nptype(x.dtype)).max)
        mask = ms.numpy.expand_dims(mask, 1)

        # Caculate the global query
        alpha_weight = (ops.mul(query, self.weight_alpha) * self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = ops.softmax(alpha_weight, axis = -1)
        global_query = query * alpha_weight
        global_query = ops.sum(global_query, dim=1) 

        # Model the interaction between global query vector and the key vector
        repeat_global_query = ops.ExpandDims()(global_query, 1)
        repeat_shape = (1, n, 1)
        repeat_global_query = ops.Tile()(repeat_global_query, repeat_shape)
        p = repeat_global_query * key
        beta_weight = (ops.mul(p, self.weight_beta) * self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = ops.softmax(beta_weight, axis = -1)
        global_key = p * beta_weight
        global_key = ops.sum(global_key, dim=1)

        # key-value
        batch_size, num_items, dim = value.shape
        global_key_expanded = ops.ExpandDims()(global_key, 1)
        global_key_tiled = ops.Tile()(global_key_expanded, (1, num_items, 1))
        key_value_interaction = global_key_tiled * value
        
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result

if __name__ == '__main__':
    model = Fastformer(dim = 3, decode_dim = 8)
    x = ops.randn(4, 6, 3)
    mask = ops.ones((1, 8), dtype=ms.bool_)
    result = model(x, mask)
    print(result[0])