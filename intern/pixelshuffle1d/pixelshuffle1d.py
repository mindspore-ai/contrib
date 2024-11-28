import mindspore
import mindspore.nn as nn

# "long" and "short" denote longer and shorter samples

class PixelShuffle1D(nn.Cell):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def construct(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = int(short_channel_len // self.upscale_factor)
        long_width = int(self.upscale_factor * short_width)

        x = x.contiguous().view(batch_size, int(self.upscale_factor), long_channel_len, short_width)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class PixelUnshuffle1D(nn.Cell):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def construct(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = int(long_channel_len * self.downscale_factor)
        short_width = int(long_width // self.downscale_factor)

        x = x.contiguous().view(batch_size, long_channel_len, short_width, int(self.downscale_factor))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, short_channel_len, short_width)
        return x