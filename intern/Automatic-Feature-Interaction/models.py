# coding=utf-8

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.common.initializer as init


# AUTOINT: AUTOMATIC FEATURE INTERACTION LEARNING
class AutoIN(nn.Cell):
    """'auto In model."""

    def __init__(self, cal_size, values_size, embeding_size, cal_num=1, value_num=2, dropout=0.5):
        super(AutoIN, self).__init__()
        self.cal_embedinds = nn.Embedding(vocab_size=cal_size,
                                          embedding_size=embeding_size,
                                          embedding_table=init.initializer(init.XavierNormal(gain=1.4142135623730951),
                                                                           [cal_size, embeding_size],
                                                                           mindspore.float32))

        self.value_embedings = nn.Embedding(vocab_size=values_size,
                                            embedding_size=embeding_size,
                                            embedding_table=init.initializer(init.XavierNormal(gain=1.4142135623730951),
                                                                             [values_size, embeding_size],
                                                                             mindspore.float32))

        self.MultiHead_1 = MultiHead(model_dim=embeding_size, output_dim=embeding_size // 2, num_head=8, dropout=0.5)
        self.MultiHead_2 = MultiHead(model_dim=embeding_size // 2, output_dim=embeding_size // 2, num_head=4,
                                     dropout=0.5)
        self.MultiHead_3 = MultiHead(model_dim=embeding_size // 2, output_dim=embeding_size // 2, num_head=4,
                                     dropout=0.5)
        num_output = value_num + cal_num
        self.fc_final = nn.Dense(num_output * embeding_size // 2, 1, has_bias=False)
        self.dropout = dropout

        self.values_size = values_size
        self.embeding_size = embeding_size
        self.cal_size = cal_size

    def construct(self, cal_index, value_data, value_index):
        batch_size = value_data.shape[0]
        cal_dim = self.cal_embedinds(cal_index)
        value_dim = self.value_embedings(value_index)
        value_dim = value_data * value_dim
        data_dim = ops.cat([value_dim, cal_dim], 1)

        # attention base on transformer structure. See NLP BERT module.
        print(data_dim.shape)
        data_dim = self.MultiHead_1(data_dim)
        data_dim = ops.relu(data_dim)
        print(data_dim.shape)
        data_dim = self.MultiHead_2(data_dim)
        data_dim = ops.relu(data_dim)
        print(data_dim.shape)
        data_dim = self.MultiHead_3(data_dim)
        data_dim = ops.relu(data_dim)
        print(data_dim.shape)
        data_dim = data_dim.view(batch_size, -1)
        print(data_dim.shape)
        # fc and dropout.
        data_dim = ops.dropout(data_dim)
        output = self.fc_final(data_dim)
        output = ops.sigmoid(output)
        return output


class MultiHead(nn.Cell):
    def __init__(self, model_dim=256, output_dim=128, num_head=8, dropout=0.5):
        super(MultiHead, self).__init__()
        self.dim_per_head = model_dim // num_head
        self.num_head = num_head
        self.linear_q = nn.Dense(model_dim, self.dim_per_head * num_head)
        self.linear_k = nn.Dense(model_dim, self.dim_per_head * num_head)
        self.linear_v = nn.Dense(model_dim, self.dim_per_head * num_head)

        self.product_attention = ScaledDotProductAttention(dropout)
        self.fc = nn.Dense(model_dim, output_dim)
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm([model_dim])
        # 设置 gamma 和 beta 为不可学习
        self.layer_norm.gamma.requires_grad = False
        self.layer_norm.beta.requires_grad = False

    def construct(self, x):
        residual = x
        batch_size = x.shape[0]
        # linear projection
        key = self.linear_k(x)
        value = self.linear_v(x)
        query = self.linear_q(x)

        # reshape
        key = key.view(batch_size * self.num_head, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_head, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_head, -1, self.dim_per_head)

        # attention
        context = self.product_attention(query, key, value, 8)
        # concat
        context = context.view(residual.shape)
        # residual
        context += residual
        # layer normal
        context = self.layer_norm(context)
        # fc
        context = self.fc(context)

        return context


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, dropout=0.5):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = dropout

    def construct(self, q, k, v, scale=None):
        attention = ops.bmm(q, k.swapaxes(1, 2))  # Q*K
        if scale:
            attention = attention * scale
        attention = ops.softmax(attention, axis=2)  # softmax
        attention = ops.dropout(attention)  # dropout
        context = ops.bmm(attention, v)  # attention
        return context


if __name__ == '__main__':
    cal_size = 4
    values_size = 10
    embeding_size = 256

    cal_num = 2
    value_num = 3

    model = AutoIN(cal_size, values_size, embeding_size, cal_num, value_num, dropout=0.5)
    # print(model)
    cal_index = Tensor([1, 3], mindspore.int32).unsqueeze(0)
    value_data = Tensor([[0.5], [0.1], [0.1]]).unsqueeze(0)
    value_index = Tensor([0, 1, 5]).unsqueeze(0)
    # check
    assert cal_index.shape[1] == cal_num
    assert value_data.shape[1] == value_num
    assert value_index.shape[1] == value_num

    r = model(cal_index, value_data, value_index)
    print(r)
