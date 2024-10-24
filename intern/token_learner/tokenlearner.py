import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class SpatialAttention(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1, has_bias=True, pad_mode="valid"),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
        self.sgap = nn.AvgPool2d(2)

    def construct(self, x):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        
        mx = ops.max(x, 1)[0].unsqueeze(1)
        avg = ops.mean(x, 1).unsqueeze(1)
        combined = ops.cat([mx, avg], axis=1)
        fmap = self.conv(combined)
        weight_map = ops.sigmoid(fmap)
        out = (x * weight_map).mean(axis=(-2, -1))
        
        return out, x * weight_map

class TokenLearner(nn.Cell):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.CellList([SpatialAttention() for _ in range(S)])
        
    def construct(self, x):
        B, _, _, C = x.shape
        Z = ops.zeros((B, self.S, C), mindspore.float32)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x) # [B, C]
            Z[:, i, :] = Ai
        return Z
    
class TokenFuser(nn.Cell):
    def __init__(self, H, W, C, S) -> None:
        super().__init__()
        self.projection = nn.Dense(S, S, has_bias=False)
        self.Bi = nn.Dense(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S
        
    def construct(self, y, x):
        B, S, C = y.shape
        B, H, W, C = x.shape
        
        Y = self.projection(y.view(B, C, S)).view(B, S, C)
        Bw = ops.sigmoid(self.Bi(x)).view(B, H*W, S) # [B, HW, S]
        BwY = ops.matmul(Bw, Y)
        
        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H*W, C)
        
        out = (BwY + xj).view(B, H, W, C)
        
        return out 


if __name__ == "__main__":
    mhsa = nn.MultiheadAttention(3, 1)
    tklr = TokenLearner(S=8)
    tkfr = TokenFuser(32, 32, 3, S=8)

    x = ops.rand(512, 32, 32, 3)

    y = tklr(x)
    y = y.view(8, 512, 3)
    y, _ = mhsa(y, y, y) # ignore attn weights
    y = y.view(512, 8, 3)

    out = tkfr(y, x)
    print (out.shape)