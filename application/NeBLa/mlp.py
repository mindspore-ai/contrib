import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


ms.set_context(device_target='Ascend', device_id=0)


# MLP part from the generation module
class NeRF(nn.Cell):
    def __init__(self, D=8, W=128, input_ch=42, input_ch_views=3, output_ch=1, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        self.skips = skips
        self.first_pts_linear = nn.Dense(input_ch, W)
        self.image_linear = nn.Dense(128, W)  # 128
        self.pts_linears = nn.CellList(
            [nn.Dense(W, W) if i not in self.skips else nn.Dense(W * 2, W) for i in range(D - 1)])
        self.output_linear = nn.Dense(W, output_ch)

    def construct(self, x, c):
        x = self.first_pts_linear(x)
        c = ops.Tile()(c[:, None, :], (1, x.shape[1], 1))
        h = x + self.image_linear(c)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = ops.ReLU()(h)
            if i + 1 in self.skips:
                h = ops.Concat(axis=-1)((x + self.image_linear(c), h))

        outputs = self.output_linear(h)

        return ops.Sigmoid()(outputs)
    

def main():

    D = 8
    W = 128
    input_ch = 42
    input_ch_views = 3
    output_ch = 16
    skips = [4]
    batch_size = 100
    num_samples = 10  

    x = ops.randn((batch_size, num_samples, input_ch))
    c = ops.randn((batch_size, W))

    model = NeRF(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views, output_ch=output_ch, skips=skips)
    output = model(x, c)

    print(f"Input shape: {x.shape}")
    print(f"Conditioning shape: {c.shape}")
    print(f"Output shape: {output.shape}")
    
    
if __name__ == "__main__":
    main()