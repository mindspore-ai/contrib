import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn import Cell

class IDN(Cell):
    def __init__(self, in_channel, style_dim, content_channel=1):
        super().__init__()
        
        # 使用MSAdapter的spectral_norm（需确认是否支持，否则需自定义）
        self.style_mu_conv = nn.SequentialCell(
            nn.Conv2d(in_channel, in_channel, 4, 2, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.1),
            nn.Conv2d(in_channel, in_channel, 4, 2, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.to_style = nn.Dense(in_channel*2, style_dim)

        self.to_content = nn.SequentialCell(
            nn.Conv2d(in_channel, in_channel, 3, 1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.1),
            nn.Conv2d(in_channel, content_channel, 3, 1, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.1),
            nn.AdaptiveAvgPool2d(16)
        )

    def construct(self, feat):
        b, c, _, _ = feat.shape

        style_mu = self.style_mu_conv(feat)

        feat_no_style_mu = feat - style_mu
        style_sigma = ops.std(feat_no_style_mu.view(b, c, -1), axis=-1)
        feat_content = feat_no_style_mu / style_sigma.view(b,c,1,1)
        style = self.to_style( ops.cat([style_mu.view(b,-1), style_sigma.view(b,-1)], axis=1) )
        content = self.to_content(feat_content)
        
        return content, style
class FeatMapTransfer(Cell):
    def rescale(self, tensor, range=(0, 1)):
        return ((tensor - tensor.min()) / (tensor.max() - tensor.min()))*(range[1]-range[0]) + range[0]

    def __init__(self, hw=64):
        super().__init__()
        self.ad_pool = nn.AdaptiveAvgPool2d(hw)
        self.feat_pool = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.AdaptiveAvgPool2d(hw//8),
        )
        
    def construct(self, style_feat, style_skt, content_skt):
        style_feat = self.ad_pool(style_feat)

        style_skt = self.rescale(self.ad_pool(style_skt))
        content_skt = self.rescale(self.ad_pool(content_skt))

        edge_feat = style_feat * style_skt
        plain_feat = style_feat * (1-style_skt)

        edge_feat = self.feat_pool(edge_feat).tile((1,1,8,8))
        plain_feat = self.feat_pool(plain_feat).tile((1,1,8,8))

        return edge_feat*content_skt + plain_feat*(1-content_skt)
class DualMaskInjection(Cell):
    def __init__(self, in_channels):
        super().__init__()
        
        self.weight_a = ms.Parameter(ops.ones((1, in_channels, 1, 1)) * 1.1)
        self.weight_b = ms.Parameter(ops.ones((1, in_channels, 1, 1)) * 0.9)

        self.bias_a = ms.Parameter(ops.zeros((1, in_channels, 1, 1)) + 0.01)
        self.bias_b = ms.Parameter(ops.zeros((1, in_channels, 1, 1)) + 0.01)

    def construct(self, feat, mask):
        feat_a = self.weight_a * feat * mask + self.bias_a
        feat_b = self.weight_b * feat * (1-mask) + self.bias_b
        return feat_a + feat_b
def test_main():
    # 设置运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    # 1. 初始化模型
    idn = IDN(in_channel=3, style_dim=256)
    feat_transfer = FeatMapTransfer()
    dual_mask = DualMaskInjection(in_channels=3)
    
    # 2. 生成测试数据
    input_feat = ms.Tensor(np.random.randn(8, 3, 64, 64), dtype=ms.float32)
    style_skt = ms.Tensor(np.random.rand(8, 1, 64, 64), dtype=ms.float32)
    content_skt = ms.Tensor(np.random.rand(8, 1, 64, 64), dtype=ms.float32)
    
    # 3. 前向传播测试
    content_feat, style_feat = idn(input_feat)
    print(f"IDN输出形状: Content {content_feat.shape}, Style {style_feat.shape}")
    
    transferred_feat = feat_transfer(input_feat, style_skt, content_skt)
    print(f"FeatMapTransfer输出形状: {transferred_feat.shape}")
    
    mask = ms.Tensor(np.random.rand(8, 3, 64, 64), dtype=ms.float32)
    dual_output = dual_mask(transferred_feat, mask)
    print(f"DualMaskInjection输出形状: {dual_output.shape}")

if __name__ == "__main__":
    test_main()