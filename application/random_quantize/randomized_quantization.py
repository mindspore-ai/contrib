import mindspore as ms
from mindspore.nn import Cell

class RandomizedQuantizationAugModule(Cell):
    def __init__(
            self, region_num,
            collapse_to_val = 'inside_random',
            spacing='random',
            transforms_like=False,
            p_random_apply_rand_quant = 1
            ):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        self.transforms_like = transforms_like
        self.p_random_apply_rand_quant = p_random_apply_rand_quant


    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.shape # one batch img
        min_val, max_val = x.view(C, -1).min(1), x.view(C, -1).max(1)
        # min, max over batch size, spatial dimension
        total_region_percentile_number = (ms.ops.ones(C) * (self.region_num - 1)).to(ms.float32)
        return min_val, max_val, total_region_percentile_number


    def construct(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1
        if self.p_random_apply_rand_quant != 1:
            x_orig = x
        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c
            x = x.view(C, H, W)
        else:
            C, H, W = x.shape

        # -> (C), (C), (C)
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x)

        # region percentiles for each channel
        if self.spacing == "random":
            region_percentiles = ms.ops.rand(int(total_region_percentile_number_per_channel.sum()))
        elif self.spacing == "uniform":
            region_percentiles = ms.ops.tile(
                ms.ops.arange(1/(total_region_percentile_number_per_channel[0] + 1),
                              1,
                              step=1/(total_region_percentile_number_per_channel[0]+1)),
                              (C,)
                              )
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])
        # ordered region ends
        region_percentiles_pos = (
            region_percentiles_per_channel * (max_val - min_val).view(C, 1)
            + min_val.view(C, 1)
            ).view(C, -1, 1, 1)
        ordered_region_right_ends_for_checking = ms.ops.cat(
            [region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON],
            axis=1
            ).sort(1)[0]
        ordered_region_right_ends = ms.ops.cat(
            [region_percentiles_pos,
             max_val.view(C, 1, 1, 1)+1e-6],
             axis=1
             ).sort(1)[0]
        ordered_region_left_ends = ms.ops.cat(
            [min_val.view(C, 1, 1, 1), region_percentiles_pos],
            axis=1
            ).sort(1)[0]
        # ordered middle points
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2
        # associate region id
        is_inside_each_region = (
            (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking).to(ms.int32) *
            (x.view(C, 1, H, W) >= ordered_region_left_ends).to(ms.int32)
        )
        # -> (C, self.region_num, H, W); boolean -> int

        assert (is_inside_each_region.sum(1) == 1).all()
        # sanity check: each pixel falls into one sub_range
        associated_region_id = ms.ops.argmax(is_inside_each_region.int(), dim=1, keepdim=True)
        # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            # middle points as the proxy for all values in corresponding regions
            proxy_vals = ms.ops.gather_elements(
                ordered_region_mid.broadcast_to((-1, -1, H, W)),
                1,
                associated_region_id
                )[:,0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'inside_random':
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = ms.ops.rand(
                int((total_region_percentile_number_per_channel + 1).sum())
                )
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape(
                [-1, self.region_num]
                )
            ordered_region_rand = (
                ordered_region_left_ends
                + proxy_percentiles_per_channel.view(C, -1, 1, 1)
                * (ordered_region_right_ends - ordered_region_left_ends)
                )
            proxy_vals = ms.ops.gather_elements(
                ordered_region_rand.broadcast_to((-1, -1, H, W)),
                1,
                associated_region_id
                )[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            proxy_vals = ms.ops.zeros_like(x)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                x = ms.ops.where(ms.ops.rand([B,1,1,1]) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                x = ms.ops.where(ms.ops.rand([C,1,1]) < self.p_random_apply_rand_quant, x, x_orig)

        return x


if __name__ == "__main__":
    batch_size, channels, height, width = 2, 3, 32, 64
    region_num = 5
    collapse_to_val_options = ['middle', 'inside_random', 'all_zeros']
    spacing_options = ['random', 'uniform']
    transforms_like_options = [False, True]
    p_random_apply_rand_quant_options = [1, 0.5]

    x = ms.ops.rand(batch_size, channels, height, width)

    for collapse_to_val in collapse_to_val_options:
        for spacing in spacing_options:
            for transforms_like in transforms_like_options:
                for p_random_apply_rand_quant in p_random_apply_rand_quant_options:
                    module = RandomizedQuantizationAugModule(
                        region_num=region_num,
                        collapse_to_val=collapse_to_val,
                        spacing=spacing,
                        transforms_like=transforms_like,
                        p_random_apply_rand_quant=p_random_apply_rand_quant
                    )

                    if transforms_like:
                        x_input = x[0]
                    else:
                        x_input = x
                    print(
                        f"Testing with collapse_to_val={collapse_to_val}, "
                        f"spacing={spacing}, transforms_like={transforms_like}, "
                        f"p_random_apply_rand_quant={p_random_apply_rand_quant}"
                    )
                    output = module(x_input)
                    print("The Output shape is:", output.shape)
                    assert output.shape == x_input.shape, "The output shape is not correct!"

    print("All tests passed!")
