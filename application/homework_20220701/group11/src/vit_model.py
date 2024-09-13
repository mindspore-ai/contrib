# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Vision Transformer Backbone."""

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P

from mindvision.check_param import Validator
from mindvision.classification.models.blocks import PatchEmbedding, TransformerEncoder
from mindvision.classification.utils import init


class ViT(nn.Cell):
    """
    Vision Transformer architecture implementation.

    Args:
        image_size (int): Input image size. Default: 224.
        input_channels (int): The number of input channel. Default: 3.
        patch_size (int): Patch size of image. Default: 16.
        embed_dim (int): The dimension of embedding. Default: 768.
        num_layers (int): The depth of transformer. Default: 12.
        num_heads (int): The number of attention heads. Default: 12.
        mlp_dim (int): The dimension of MLP hidden layer. Default: 3072.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention layer. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.LayerNorm.
        pool (str): The method of pooling. Default: 'cls'.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ViT(224, 3, 32, 768, 12, 12, 3072, 0.1)
    """

    def __init__(self,
                 image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm,
                 pool: str = 'mean') -> None:
        super(ViT, self).__init__()

        Validator.check_string(pool, ["cls", "mean", "relu"], "pool type")

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=input_channels)
        num_patches = self.patch_embedding.num_patches

        if pool == "cls":
            self.cls_token = init(init_type=Normal(sigma=1.0),
                                  shape=(1, 1, embed_dim),
                                  dtype=ms.float32,
                                  name='cls',
                                  requires_grad=True)
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches + 1, embed_dim),
                                      dtype=ms.float32,
                                      name='pos_embedding',
                                      requires_grad=True)
            self.concat = P.Concat(axis=1)
        elif pool == 'mean':
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches, embed_dim),
                                      dtype=ms.float32,
                                      name='pos_embedding',
                                      requires_grad=True)
            self.mean = P.ReduceMean(keep_dims=False)

        else:
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches, embed_dim),
                                      dtype=ms.float32,
                                      name='pos_embedding',
                                      requires_grad=True)
            self.relu = nn.ReLU()

        self.pool = pool
        self.pos_dropout = nn.Dropout(keep_prob)
        self.norm = norm((embed_dim,))
        self.tile = P.Tile()
        self.transformer = TransformerEncoder(dim=embed_dim,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              keep_prob=keep_prob,
                                              attention_keep_prob=attention_keep_prob,
                                              drop_path_keep_prob=drop_path_keep_prob,
                                              activation=activation,
                                              norm=norm)

    def construct(self, x):
        """ViT construct."""
        x = self.patch_embedding(x)

        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (x.shape[0], 1, 1))
            x = self.concat((cls_tokens, x))
            x += self.pos_embedding
        else:
            x += self.pos_embedding
        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        elif self.pool == "mean":
            x = self.mean(x, (1,))  # (1,) or (1,2)
            # x = self.relu(x)
        else:
            x = self.relu(x)
        return x


if __name__ == '__main__':
    model = ViT(pool='mean')
    print(model.trainable_params())
