"""
transformer结构
"""
from multi_model.nlp.bert_model import MytransformerDecoder
from multi_model.vision.vit_model import ViT
from mindspore.ops import operations as P
from mindspore import nn


class Transformer(nn.Cell):
    """构建transformer"""

    def __init__(self, image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 vocab_size: int = 10310,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm,
                 pool: str = 'relu',
                 keep_prob: float = 1.0):
        super(Transformer, self).__init__()
        self.encoder = ViT(image_size=image_size, input_channels=input_channels, patch_size=patch_size,
                           embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim, pool=pool)
        self.decoder = MytransformerDecoder(dim=embed_dim,
                                            num_layers=num_layers,
                                            num_heads=num_heads,
                                            mlp_dim=mlp_dim,
                                            vocab_size=vocab_size,
                                            attention_keep_prob=1.0,
                                            drop_path_keep_prob=1.0,
                                            activation=activation,
                                            norm=norm)
        # print(type(activation))
        self.dropout = nn.Dropout(keep_prob)
        self.mean = P.ReduceMean()
        # self.dense = nn.Dense(embed_dim*((image_size**2)//(patch_size**2)), 1, has_bias=True)
        self.dense = nn.Dense(embed_dim, 1, has_bias=True)
        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()

    def construct(self, encoder_x, decoder_x, seq_length):
        """模型计算过程"""
        encoder_out = self.encoder(encoder_x)
        decoder_out = self.decoder(encoder_out, decoder_x, seq_length)
        # b, n, dim = decoder_out.shape
        # if self.train:
        #     x = self.dropout(encoder_out)
        decoder_out = self.mean(decoder_out, (1,))
        print('encoder_out:', decoder_out.shape)
        # decoder_out = self.reshape(decoder_out, (b, n * dim))
        x = self.dense(decoder_out)
        x = self.sigmoid(x)
        # if self.train:
        #     x = self.dropout(decoder_out)
        return x


if __name__ == "__main__":
    mymodel = Transformer()
    print(mymodel)
