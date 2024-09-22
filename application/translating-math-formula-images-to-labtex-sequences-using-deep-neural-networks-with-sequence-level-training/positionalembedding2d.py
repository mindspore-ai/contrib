import math
import mindspore as ms
from mindspore import ops

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
    pe = ops.zeros((length, d_model))
    position = ops.arange(length).expand_dims(1)
    div_term = ops.exp((ops.arange(0, d_model, 2, dtype=ms.float32) *
                        -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = ops.sin(position.float() * div_term)
    pe[:, 1::2] = ops.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = ops.zeros((d_model, height, width))
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = ops.exp(ops.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = ops.arange(0., width).expand_dims(1)
    pos_h = ops.arange(0., height).expand_dims(1)
    pe[0:d_model:2, :, :] = ops.sin(pos_w * div_term).transpose(0, 1).expand_dims(1).reshape(d_model, height, 1)
    pe[1:d_model:2, :, :] = ops.cos(pos_w * div_term).transpose(0, 1).expand_dims(1).reshape(d_model, height, 1)
    pe[d_model::2, :, :] = ops.sin(pos_h * div_term).transpose(0, 1).expand_dims(2).reshape(d_model, 1, width)
    pe[d_model + 1::2, :, :] = ops.cos(pos_h * div_term).transpose(0, 1).expand_dims(2).reshape(d_model, 1, width)

    return pe