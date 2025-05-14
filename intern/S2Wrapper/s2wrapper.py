import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
import numpy as np


def split_chessboard(x, num_split):
    """
    Split the input tensor into num_split**2 sub-blocks and concatenate them along the batch dimension.

    Args:
        x (Tensor): Input tensor of shape (b, c, h, w)
        num_split (int): Number of splits

    Returns:
        Tensor: The split tensor
    """
    b, c, h, w = x.shape
    assert h % num_split == 0 and w % num_split == 0, "Input size must be divisible by num_split"
    x = x.view(b, c, num_split, h // num_split, num_split, w // num_split)
    x = x.permute(2, 4, 0, 1, 3, 5)
    x = x.reshape(num_split * num_split * b, c, h // num_split, w // num_split)
    return x


def merge_chessboard(x, num_split):
    """
    Merge the split sub-blocks back to the original shape.

    Args:
        x (Tensor): Tensor containing num_split**2 sub-blocks
        num_split (int): Number of splits

    Returns:
        Tensor: The merged tensor
    """
    b, c, h, w = x.shape
    assert b % (num_split ** 2) == 0, "Batch size must be divisible by num_split**2"
    x = x.view(num_split, num_split, b // (num_split ** 2), c, h, w)
    x = x.permute(2, 3, 0, 4, 1, 5)
    x = x.reshape(b // (num_split ** 2), c, h * num_split, w * num_split)
    return x


def batched_forward(model, x, batch_size=-1):
    """
    Forward input in batches.

    Args:
        model (nn.Cell): Model
        x (Tensor): Input tensor
        batch_size (int): Batch size, -1 means no split

    Returns:
        Tensor: Model output
    """
    if batch_size == -1:
        return model(x)
    split = ops.Split(0, batch_size)
    x_batched = split(x)
    outs = [model(x) for x in x_batched]
    return ops.Concat(0)(outs)


def forward(model, input_tensor, scales=None, img_sizes=None, max_split_size=None,
            resize_output_to_idx=0, num_prefix_token=0, output_shape='bnc',
            split_forward=False):
    """
    Multi-scale forward function.

    Args:
        model (nn.Cell): Model
        input_tensor (Tensor): Input tensor
        scales (list): List of scaling factors
        img_sizes (list): List of image sizes
        max_split_size (int): Maximum split size
        resize_output_to_idx (int): Index to resize output
        num_prefix_token (int): Number of prefix tokens
        output_shape (str): Output shape, 'bnc' or 'bchw'
        split_forward (bool): Whether to use batched forward

    Returns:
        Tensor: Processed output tensor
    """
    if input_tensor.ndim != 4:
        raise ValueError("Input image must be of shape BxCxHxW")
    if input_tensor.shape[2] != input_tensor.shape[3]:
        raise ValueError("Only square images are supported")
    if output_shape not in ['bnc', 'bchw']:
        raise ValueError("Output shape must be 'bnc' or 'bchw'")
    if output_shape == 'bchw' and num_prefix_token != 0:
        raise ValueError("Prefix tokens are not supported for ConvNet")

    b, c, input_size, _ = input_tensor.shape

    if scales is None and img_sizes is None:
        raise ValueError("Either scales or img_sizes must be specified")
    img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

    max_split_size = max_split_size or input_size
    num_splits = [math.ceil(size / max_split_size) for size in img_sizes]
    input_multiscale = []

    def resize_fn(x, size):
        return ops.interpolate(x, size=(size, size), mode="bilinear")

    for size, num_split in zip(img_sizes, num_splits):
        x = resize_fn(input_tensor.astype(mstype.float32), size)
        x = x.astype(input_tensor.dtype)
        x = split_chessboard(x, num_split=num_split)
        input_multiscale.append(x)

    outs_multiscale = [
        batched_forward(model, x, b) if split_forward else model(x)
        for x in input_multiscale
    ]

    if num_prefix_token > 0:
        outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
        outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]

    if output_shape == 'bnc':
        for i, out in enumerate(outs_multiscale):
            h = int(out.shape[1] ** 0.5)
            outs_multiscale[i] = out.view(b, h, h, -1).permute(0, 3, 1, 2)

    outs_multiscale = [
        merge_chessboard(out, num_split=num_split)
        for num_split, out in zip(num_splits, outs_multiscale)
    ]

    output_size = outs_multiscale[resize_output_to_idx].shape[-2]
    out = ops.Concat(1)([
        resize_fn(out.astype(mstype.float32), output_size).astype(out.dtype)
        for out in outs_multiscale
    ])

    if output_shape == 'bnc':
        out = out.permute(0, 2, 3, 1).view(b, -1, out.shape[1])

    if num_prefix_token > 0:
        outs_prefix_multiscale = [
            ops.Stack(0)(out.split(b, dim=0)).mean(axis=0)
            for out in outs_prefix_multiscale
        ]
        out_prefix_multiscale = ops.Concat(-1)(outs_prefix_multiscale)
        out = ops.Concat(1)([out_prefix_multiscale, out])

    return out


if __name__ == '__main__':
    # Set random seed
    ms.set_seed(42)

    # Define a simple model
    class SimpleModel(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=0)

        def construct(self, x):
            return self.conv(x)

    # Create example data
    batch_size = 2
    channels = 3
    image_size = 256

    # Generate random input images
    input_images = Tensor(
        np.random.randn(batch_size, channels, image_size, image_size),
        dtype=mstype.float32
    )

    # Initialize model
    model = SimpleModel()

    # Set multi-scale parameters
    scales = [0.5, 1.0, 2.0]
    max_split_size = 128

    # Forward with S2Wrapper
    output = forward(
        model=model,
        input_tensor=input_images,
        scales=scales,
        max_split_size=max_split_size,
        output_shape='bchw',
        split_forward=True
    )

    # Print results
    print("Input image shape:", input_images.shape)
    print("Output shape:", output.shape)
    print("Scales used:", scales)
    print("Max split size:", max_split_size)

    # Print statistics
    print("\nStatistics:")
    print("Input mean:", input_images.mean().asnumpy())
    print("Input std:", input_images.std().asnumpy())
    print("Output mean:", output.mean().asnumpy())
    print("Output std:", output.std().asnumpy())