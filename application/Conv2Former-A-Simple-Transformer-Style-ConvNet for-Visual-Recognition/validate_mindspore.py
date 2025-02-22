import os
import csv
import glob
import time
import logging
import argparse
from typing import Optional, Set
from collections import OrderedDict
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import context, Tensor
from mindspore import Tensor, Parameter
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer, TruncatedNormal, Constant
from collections import OrderedDict

class MLP(nn.Cell):
    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=1, has_bias=True)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, kernel_size=3, pad_mode='pad', padding=1, group=dim * mlp_ratio, has_bias=True)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1, has_bias=True)
        self.act = nn.GELU()

    def construct(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x

class SpatialAttention(nn.Cell):
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super(SpatialAttention, self).__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=1, has_bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, pad_mode='pad', padding=kernel_size//2, group=dim, has_bias=True)
        )
        self.v = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, has_bias=True)

    def construct(self, x):
        x = self.norm(x)        
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.keep_prob = 1.0 - drop_prob
        self.rand = ops.UniformReal()
        self.floor = ops.Floor()
        self.div = ops.Div()

    def construct(self, x):
        if self.training and self.drop_prob > 0.0:
            random_tensor = self.rand((x.shape[0],) + (1,) * (x.ndim - 1)) 
            random_tensor += self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = self.div(x, self.keep_prob) * random_tensor
        return x

class Block(nn.Cell):
    def __init__(self, index, dim, kernel_size, num_head, window_size=14, mlp_ratio=4., drop_path=0.):
        super(Block, self).__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        
        self.layer_scale_1 = Parameter(
            Tensor(layer_scale_init_value * ops.ones((dim), ms.float32)), requires_grad=True)
        self.layer_scale_2 = Parameter(
            Tensor(layer_scale_init_value * ops.ones((dim), ms.float32)), requires_grad=True)

    def construct(self, x):
        x = x + self.drop_path(self.layer_scale_1.reshape(1, -1, 1, 1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.reshape(1, -1, 1, 1) * self.mlp(x))
        return x

class Conv2Former(nn.Cell):
    def __init__(self, kernel_size, img_size=224, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], window_sizes=[14, 14, 14, 7],
                 mlp_ratios=[4, 4, 4, 4], num_heads=[2, 4, 10, 16], layer_scale_init_value=1e-6, 
                 head_init_scale=1., drop_path_rate=0., drop_rate=0.):
        super(Conv2Former, self).__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.downsample_layers = nn.CellList()
        stem = nn.SequentialCell(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=2, stride=2, has_bias=False),
        )
        self.downsample_layers.append(stem)
        
        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.SequentialCell(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=stride, stride=stride, has_bias=True)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()
        dp_rates = [x for x in ops.linspace(Tensor(0, ms.float32), Tensor(drop_path_rate, ms.float32), sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.SequentialCell([
                Block(cur+j, dims[i], kernel_size, num_heads[i], window_sizes[i], 
                mlp_ratios[i], dp_rates[cur+j]) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.SequentialCell(
            nn.Conv2d(dims[-1], 1280, kernel_size=1, has_bias=True),
            nn.GELU(),
            LayerNorm(1280, eps=1e-6, data_format="channels_first")
        )
        self.pred = nn.Dense(1280, num_classes)
        self._init_weights()

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, LayerNorm):
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
                cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))

    def construct_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x.mean(axis=(-2, -1))  # Global average pooling

    def construct(self, x):
        x = self.construct_features(x)
        x = self.pred(x)
        return x

class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = Parameter(initializer('ones', normalized_shape), name='weight')
        self.bias = Parameter(initializer('zeros', normalized_shape), name='bias')
        self.eps = eps
        self.data_format = data_format
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()

    def construct(self, x):
        if self.data_format == "channels_last":
            return nn.LayerNorm((x.shape[-1],), epsilon=self.eps)(x)
        else:
            u = self.reduce_mean(x, 1)
            s = self.reduce_mean(self.square(x - u), 1)
            x = (x - u) / self.sqrt(s + self.eps)
            x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(1, -1, 1, 1)
            return x

# Model variants (register_model decorators can be removed or adapted as needed)


def conv2former_n(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=7, dims=[64, 128, 256, 512], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 8, 2], **kwargs)
    return model


def conv2former_t(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[72, 144, 288, 576], mlp_ratios=[4, 4, 4, 4], depths=[3, 3, 12, 3], **kwargs)
    return model


def conv2former_s(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[72, 144, 288, 576], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 32, 4], **kwargs)
    return model


def conv2former_b(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[96, 192, 384, 768], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    return model


def conv2former_b_22k(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=7, dims=[96, 192, 384, 768], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    return model


def conv2former_l(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[128, 256, 512, 1024], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 48, 4], **kwargs)
    return model











_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='MindSpore ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='conv2former_b',
                    help='model architecture (default: conv2former_b)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image size')
parser.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percentage')
parser.add_argument('--mean', type=float, nargs='+', default=[0.485, 0.456, 0.406],
                    help='Mean pixel value')
parser.add_argument('--std', type=float, nargs='+', default=[0.229, 0.224, 0.225],
                    help='Std deviation')
parser.add_argument('--interpolation', default='bilinear', type=str,
                    help='Image resize interpolation type')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number of classes in dataset')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to checkpoint')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUs to use (currently not supported)')
parser.add_argument('--log-freq', default=10, type=int,
                    help='batch logging frequency')
parser.add_argument('--results-file', default='', type=str,
                    help='Output CSV file for results')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1)
    pred = pred.T
    correct = pred.equal(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(ms.float32).sum(axis=0)
        res.append(correct_k * 100.0 / batch_size)
    return res

def create_dataset(dataset_dir, training=False, **kwargs):
    """Create ImageFolder dataset"""
    data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=kwargs.get('num_workers', 4))
    return data_set

def create_loader(dataset, batch_size=32, repeat_num=1, **kwargs):
    """Create data loader"""
    # Define transformations
    mean = kwargs.get('mean', [0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = kwargs.get('std', [0.229 * 255, 0.224 * 255, 0.225 * 255])
    image_size = kwargs.get('input_size', [3, 224, 224])[1]
    
    # 新增插值方法映射字典
    interpolation_map = {
        'bilinear': Inter.BILINEAR,
        'bicubic': Inter.BICUBIC,
        'nearest': Inter.NEAREST,
        'area': Inter.AREA,
        'linear': Inter.LINEAR
    }
    interpolation = interpolation_map.get(kwargs.get('interpolation', 'bilinear').lower(), Inter.BILINEAR)

    transform = [
        vision.Decode(),
        vision.Resize(int(image_size / kwargs.get('crop_pct', 0.875)), 
                     interpolation=interpolation),  # 使用转换后的枚举值
        vision.CenterCrop(image_size),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]

    # Apply transformations
    dataset = dataset.map(operations=transform, input_columns="image")
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

def validate(args):

    # Create model
    model_map = {
        'conv2former_n': conv2former_n,
        'conv2former_t': conv2former_t,
        'conv2former_s': conv2former_s,
        'conv2former_b': conv2former_b,
        'conv2former_l': conv2former_l
    }
    model_fn = model_map.get(args.model, conv2former_b)
    model = model_fn(num_classes=args.num_classes)

    # Load checkpoint
    if args.checkpoint:
        param_dict = load_checkpoint(args.checkpoint)
        load_param_into_net(model, param_dict)

    # Dataset and loader
    dataset = create_dataset(args.data)
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        input_size=[3, args.img_size, args.img_size],
        mean=args.mean,
        std=args.std,
        interpolation=args.interpolation,
        crop_pct=args.crop_pct,
        num_workers=args.workers
    )

    # Loss function
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # Metrics
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Validate
    model.set_train(False)
    for batch_idx, data in enumerate(loader.create_dict_iterator()):
        input = data['image']
        target = data['label']

        start = time.time()
        output = model(Tensor(input, ms.float32))
        loss = criterion(output, Tensor(target, ms.int32))

        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.asnumpy(), input.shape[0])
        top1.update(acc1.asnumpy(), input.shape[0])
        top5.update(acc5.asnumpy(), input.shape[0])

        # Time measurement
        batch_time.update(time.time() - start)

        if batch_idx % args.log_freq == 0:
            _logger.info(
                'Test: [{0}/{1}]  '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})  '
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, loader.get_dataset_size(), 
                    batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    # Final results
    results = OrderedDict([
        ('top1', round(top1.avg, 4)),
        ('top5', round(top5.avg, 4)),
        ('loss', round(losses.avg, 4)),
        ('img_size', args.img_size)
    ])

    _logger.info(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(
        results['top1'], results['top5']))
    
    return results

def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    
    if args.num_gpu > 1:
        raise NotImplementedError("Multi-GPU validation not implemented")

    # Run validation
    results = validate(args)

    # Save results
    if args.results_file:
        with open(args.results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(results.keys())
            writer.writerow(results.values())

if __name__ == '__main__':
    main()