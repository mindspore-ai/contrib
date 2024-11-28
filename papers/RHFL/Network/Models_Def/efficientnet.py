"""EfficientNet model definition"""
import logging
import math
import re
from copy import deepcopy

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import (Normal, One, Uniform, Zero)
from mindspore.ops import operations as P
from mindspore.ops.composite import clip_by_value

relu = P.ReLU()
sigmoid = P.Sigmoid()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0-d6904d92.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2-cf78dc4d.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'efficientnet_b3': _cfg(
        url='', input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'efficientnet_b4': _cfg(
        url='', input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
}

_DEBUG = False

_BN_MOMENTUM_PT_DEFAULT = 0.1
_BN_EPS_PT_DEFAULT = 1e-5
_BN_ARGS_PT = dict(momentum=_BN_MOMENTUM_PT_DEFAULT, eps=_BN_EPS_PT_DEFAULT)
_BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
_BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=_BN_MOMENTUM_TF_DEFAULT, eps=_BN_EPS_TF_DEFAULT)


def _initialize_weight_goog(shape=None, layer_type='conv', bias=False):
    """
    _initialize_weight_goog
    """
    if layer_type not in ('conv', 'bn', 'fc'):
        raise ValueError('The layer type is not known, the supported are conv, bn and fc')
    if bias:
        return Zero()
    if layer_type == 'conv':
        assert isinstance(shape, (tuple, list)) and len(
            shape) == 3, 'The shape must be 3 scalars, and are in_chs, ks, out_chs respectively'
        n = shape[1] * shape[1] * shape[2]
        return Normal(math.sqrt(2.0 / n))
    if layer_type == 'bn':
        return One()
    assert isinstance(shape, (tuple, list)) and len(
        shape) == 2, 'The shape must be 2 scalars, and are in_chs, out_chs respectively'
    n = shape[1]
    init_range = 1.0 / math.sqrt(n)
    return Uniform(init_range)


def _initialize_weight_default(layer_type='conv', bias=False):
    """

    Args:
        layer_type:
        bias:

    Returns:

    """
    if layer_type not in ('conv', 'bn', 'fc'):
        raise ValueError('The layer type is not known, the supported are conv, bn and fc')
    if bias and layer_type == 'bn':
        return Zero()
    if layer_type == 'conv':
        return One()
    if layer_type == 'bn':
        return One()
    return One()


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same', bias=False):
    weight_init_value = _initialize_weight_goog(shape=(in_channels, kernel_size, out_channels))
    bias_init_value = _initialize_weight_goog(bias=True) if bias else None
    if bias:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, pad_mode=pad_mode, weight_init=weight_init_value,
                         has_bias=bias, bias_init=bias_init_value)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, pad_mode=pad_mode, weight_init=weight_init_value,
                     has_bias=bias)


def _conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='same', bias=False):
    weight_init_value = _initialize_weight_goog(shape=(in_channels, 1, out_channels))
    bias_init_value = _initialize_weight_goog(bias=True) if bias else None
    if bias:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                         padding=padding, pad_mode=pad_mode, weight_init=weight_init_value,
                         has_bias=bias, bias_init=bias_init_value)
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=padding, pad_mode=pad_mode, weight_init=weight_init_value,
                     has_bias=bias)


def _conv_group(in_channels, out_channels, group, kernel_size=3, stride=1, padding=0, pad_mode='same', bias=False):
    weight_init_value = _initialize_weight_goog(shape=(in_channels, kernel_size, out_channels))
    bias_init_value = _initialize_weight_goog(bias=True) if bias else None
    if bias:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, pad_mode=pad_mode, weight_init=weight_init_value,
                         group=group, has_bias=bias, bias_init=bias_init_value)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, pad_mode=pad_mode, weight_init=weight_init_value,
                     group=group, has_bias=bias)


def _fused_bn(channels, momentum=0.1, eps=1e-4, gamma_init=1, beta_init=0):
    return nn.BatchNorm2d(channels, eps=eps, momentum=1 - momentum, gamma_init=gamma_init, beta_init=beta_init)


def _dense(in_channels, output_channels, bias=True, activation=None):
    weight_init_value = _initialize_weight_goog(shape=(in_channels, output_channels), layer_type='fc')
    bias_init_value = _initialize_weight_goog(bias=True) if bias else None
    if bias:
        return nn.Dense(in_channels, output_channels, weight_init=weight_init_value, bias_init=bias_init_value,
                        has_bias=bias, activation=activation)
    return nn.Dense(in_channels, output_channels, weight_init=weight_init_value, has_bias=bias,
                    activation=activation)


def _resolve_bn_args(kwargs):
    bn_args = _BN_ARGS_TF.copy() if kwargs.pop('bn_tf', False) else _BN_ARGS_PT.copy()
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


def _round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels

    channels *= multiplier
    channel_min = channel_min or divisor
    new_channels = max(
        int(channels + divisor / 2) // divisor * divisor,
        channel_min)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    return [int(k) for k in ss.split('.')]


def _decode_block_str(block_str):
    """ Decode block definition string

    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat blocks,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string representation of block arguments.
    Returns:
        A list of block args (dicts)
    Raises:
        ValueError: if the string def not properly specified (TODO)
    """
    assert isinstance(block_str, str)
    ops = block_str.split('_')
    block_type = ops[0]
    ops = ops[1:]
    options = {}
    noskip = False
    for op in ops:
        if op == 'noskip':
            noskip = True
        elif op.startswith('n'):
            # activation fn
            key = op[0]
            v = op[1:]
            if v in ('re', 'r6', 'hs', 'sw'):
                print('not support')
            else:
                continue
            options[key] = value
        else:
            # all numeric options
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    act_fn = options['n'] if 'n' in options else None
    exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    fake_in_chs = int(options['fc']) if 'fc' in options else 0

    num_repeat = int(options['r'])
    # each type of block has different valid arguments, fill accordingly
    if block_type == 'ir':
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type in ('ds', 'dsa'):
        block_args = dict(
            block_type=block_type,
            dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            pw_act=block_type == 'dsa',
            noskip=block_type == 'dsa' or noskip,
        )
    elif block_type == 'er':
        block_args = dict(
            block_type=block_type,
            exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=pw_kernel_size,
            out_chs=int(options['c']),
            exp_ratio=float(options['e']),
            fake_in_chs=fake_in_chs,
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s']),
            act_fn=act_fn,
            noskip=noskip,
        )
    elif block_type == 'cn':
        block_args = dict(
            block_type=block_type,
            kernel_size=int(options['k']),
            out_chs=int(options['c']),
            stride=int(options['s']),
            act_fn=act_fn,
        )
    else:
        assert False, 'Unknown block type (%s)' % block_type

    return block_args, num_repeat


def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0, depth_trunc='ceil'):
    """ Per-stage depth scaling
    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


def _decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil'):
    """

    Args:
        arch_def:
        depth_multiplier:
        depth_trunc:

    Returns:

    """
    arch_args = []
    for _, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            stack_args.append(ba)
            repeats.append(rep)
        arch_args.append(_scale_stage_depth(stack_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


def hard_swish(x):
    """

    Args:
        x:

    Returns:

    """
    x = P.Cast()(x, ms.float32)
    y = x + 3.0
    y = clip_by_value(y, 0.0, 6.0)
    y = y / 6.0
    return x * y


class BlockBuilder(nn.Cell):
    """
    BlockBuilder
    """
    def __init__(self, builder_in_channels, builder_block_args, channel_multiplier=1.0, channel_divisor=8,
                 channel_min=None, pad_type='', act_fn=None, se_gate_fn=sigmoid, se_reduce_mid=False,
                 bn_args=None, drop_connect_rate=0., verbose=False):
        super(BlockBuilder, self).__init__()

        bn_args = _BN_ARGS_PT if bn_args is None else bn_args
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_fn = act_fn
        self.se_gate_fn = se_gate_fn
        self.se_reduce_mid = se_reduce_mid
        self.bn_args = bn_args
        self.drop_connect_rate = drop_connect_rate
        self.verbose = verbose

        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0
        self.layer = self._make_layer(builder_in_channels, builder_block_args)

    def _round_channels(self, chs):
        """

        Args:
            chs:

        Returns:

        """
        return _round_channels(chs, self.channel_multiplier, self.channel_divisor, self.channel_min)

    def _make_block(self, ba):
        """

        Args:
            ba:

        Returns:

        """
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['bn_args'] = self.bn_args
        ba['pad_type'] = self.pad_type
        ba['act_fn'] = ba['act_fn'] if ba['act_fn'] is not None else self.act_fn
        assert ba['act_fn'] is not None
        if bt == 'ir':
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_gate_fn'] = self.se_gate_fn
            ba['se_reduce_mid'] = self.se_reduce_mid
            if self.verbose:
                logging.info('  InvertedResidual %d, Args: %s', self.block_idx, str(ba))
            block = InvertedResidual(**ba)
        elif bt in ('ds', 'dsa'):
            ba['drop_connect_rate'] = self.drop_connect_rate * self.block_idx / self.block_count
            if self.verbose:
                logging.info('  DepthwiseSeparable %d, Args: %s', self.block_idx, str(ba))
            block = DepthwiseSeparableConv(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']

        return block

    def _make_stack(self, stack_args):
        """

        Args:
            stack_args:

        Returns:

        """
        blocks = []
        # each stack (stage) contains a list of block arguments
        for i, ba in enumerate(stack_args):
            if self.verbose:
                logging.info(' Block: %d', i)
            if i >= 1:
                # only the first block in any stack can have a stride > 1
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1
        return nn.SequentialCell(blocks)

    def _make_layer(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        if self.verbose:
            logging.info('Building model trunk with %d stages...', len(block_args))
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []

        for stack_idx, stack in enumerate(block_args):
            if self.verbose:
                logging.info('Stack: %d', stack_idx)
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return nn.SequentialCell(blocks)

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        return self.layer(x)


class DropConnect(nn.Cell):
    """
    DropConnect
    """
    def __init__(self, drop_connect_rate=0.):
        super(DropConnect, self).__init__()
        self.shape = P.Shape()
        self.dtype = P.DType()
        self.keep_prob = 1 - drop_connect_rate
        self.dropout = P.Dropout(keep_prob=self.keep_prob)

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        shape = self.shape(x)
        dtype = self.dtype(x)
        ones_tensor = P.Fill()(dtype, (shape[0], 1, 1, 1), 1)
        _, mask_ = self.dropout(ones_tensor)
        x = x * mask_
        return x


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """

    Args:
        inputs:
        training:
        drop_connect_rate:

    Returns:

    """
    if not training:
        return inputs
    return DropConnect(drop_connect_rate)(inputs)


class SqueezeExcite(nn.Cell):
    """
    SqueezeExcite
    """
    def __init__(self, in_chs, reduce_chs=None, act_fn=relu, gate_fn=sigmoid):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduce_chs = reduce_chs or in_chs
        self.conv_reduce = _dense(in_chs, reduce_chs, bias=True)
        self.conv_expand = _dense(reduce_chs, in_chs, bias=True)
        self.avg_global_pool = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        x_se = self.avg_global_pool(x, (2, 3))
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x_se = self.gate_fn(x_se)
        x_se = P.ExpandDims()(x_se, 2)
        x_se = P.ExpandDims()(x_se, 3)
        x = x * x_se
        return x


class DepthwiseSeparableConv(nn.Cell):
    """
    DepthwiseSeparableConv
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_fn=relu, noskip=False,
                 pw_act=False, se_ratio=0., se_gate_fn=sigmoid,
                 bn_args=None, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()

        bn_args = _BN_ARGS_PT if bn_args is None else bn_args
        assert stride in [1, 2], 'stride must be 1 or 2'
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate
        self.conv_dw = nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride, pad_mode="same",
                                 has_bias=False, group=in_chs,
                                 weight_init=_initialize_weight_goog(shape=[1, dw_kernel_size, in_chs]))
        self.bn1 = _fused_bn(in_chs, **bn_args)

        #
        if self.has_se:
            self.se = SqueezeExcite(in_chs, reduce_chs=max(1, int(in_chs * se_ratio)),
                                    act_fn=act_fn, gate_fn=se_gate_fn)
        self.conv_pw = _conv1x1(in_chs, out_chs)
        self.bn2 = _fused_bn(out_chs, **bn_args)

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        identity = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x = x + identity

        return x


class InvertedResidual(nn.Cell):
    """
    InvertedResidual
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
                 act_fn=relu, pw_kernel_size=1,
                 noskip=False, exp_ratio=1., exp_kernel_size=1, se_ratio=0.,
                 se_reduce_mid=False, se_gate_fn=sigmoid, shuffle_type=None,
                 bn_args=None, drop_connect_rate=0.):
        super(InvertedResidual, self).__init__()

        bn_args = _BN_ARGS_PT if bn_args is None else bn_args
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        self.conv_pw = _conv(in_chs, mid_chs, exp_kernel_size)
        self.bn1 = _fused_bn(mid_chs, **bn_args)

        self.shuffle_type = shuffle_type
        if self.shuffle_type is not None and isinstance(exp_kernel_size, list):
            self.shuffle = None

        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride, pad_mode="same",
                                 has_bias=False, group=mid_chs,
                                 weight_init=_initialize_weight_goog(shape=[1, dw_kernel_size, mid_chs]))
        self.bn2 = _fused_bn(mid_chs, **bn_args)

        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)),
                                    act_fn=act_fn, gate_fn=se_gate_fn)

        self.conv_pwl = _conv(mid_chs, out_chs, pw_kernel_size)
        self.bn3 = _fused_bn(out_chs, **bn_args)

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        identity = x

        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x = x + identity
        return x


class GenEfficientNet(nn.Cell):
    """
    GenEfficientNet
    """
    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=32, num_features=1280,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_fn=relu, drop_rate=0., drop_connect_rate=0.,
                 se_gate_fn=sigmoid, se_reduce_mid=False, bn_args=None,
                 head_conv='default'):
        super(GenEfficientNet, self).__init__()

        bn_args = _BN_ARGS_PT if bn_args is None else bn_args
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.act_fn = act_fn
        self.num_features = num_features

        stem_size = _round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = _conv(in_chans, stem_size, 3, stride=2)
        self.bn1 = _fused_bn(stem_size, **bn_args)
        in_chans = stem_size
        self.blocks = BlockBuilder(in_chans, block_args, channel_multiplier, channel_divisor, channel_min,
                                   pad_type, act_fn, se_gate_fn, se_reduce_mid,
                                   bn_args, drop_connect_rate, verbose=_DEBUG)
        in_chs = self.blocks.in_chs

        if not head_conv or head_conv == 'none':
            self.efficient_head = False
            self.conv_head = None
            assert in_chs == self.num_features
        else:
            self.efficient_head = head_conv == 'efficient'
            self.conv_head = _conv1x1(in_chs, self.num_features)
            self.bn2 = None if self.efficient_head else _fused_bn(self.num_features, **bn_args)
        self.global_pool = P.ReduceMean(keep_dims=True)
        self.classifier = _dense(self.num_features, self.num_classes)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.drop_out = nn.Dropout(keep_prob=1 - self.drop_rate)

    def construct(self, x):
        """

        Args:
            x:

        Returns:

        """
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x)
        x = self.blocks(x)

        if self.efficient_head:
            x = self.global_pool(x, (2, 3))
            x = self.conv_head(x)
            x = self.act_fn(x)
            x = self.reshape(self.shape(x)[0], -1)
        else:
            if self.conv_head is not None:
                x = self.conv_head(x)
                x = self.bn2(x)
            x = self.act_fn(x)
            x = self.global_pool(x, (2, 3))
            x = self.reshape(x, (self.shape(x)[0], -1))

        if self.training and self.drop_rate > 0.:
            x = self.drop_out(x)
        return self.classifier(x)


def _gen_efficientnet(channel_multiplier=1.0, depth_multiplier=1.0, num_classes=1000, **kwargs):
    """Creates an EfficientNet model.

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    num_features = _round_channels(1280, channel_multiplier, 8, None)
    model = GenEfficientNet(
        _decode_arch_def(arch_def, depth_multiplier),
        num_classes=num_classes,
        stem_size=32,
        channel_multiplier=channel_multiplier,
        num_features=num_features,
        bn_args=_resolve_bn_args(kwargs),
        act_fn=hard_swish,
        **kwargs
    )
    return model


def efficientnet_b0(num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B0 """
    model = _gen_efficientnet(
        channel_multiplier=1.0, depth_multiplier=1.0,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    return model


def efficientnet_b1(num_classes=1000, in_chans=3, **kwargs):
    """ EfficientNet-B1 """
    model = _gen_efficientnet(
        channel_multiplier=1.0, depth_multiplier=1.1,
        num_classes=num_classes, in_chans=in_chans, **kwargs)
    return model


def efficientnet():
    """

    Returns:

    """
    return efficientnet_b0(num_classes=10)
