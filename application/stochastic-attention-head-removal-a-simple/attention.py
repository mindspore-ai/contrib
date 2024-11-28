import math

import numpy
import mindspore
from mindspore import nn, ops
from head_removal import remove_head

p = 1/48
istrain = False

class MultiHeadedAttention(nn.Cell):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Dense(n_feat, n_feat)
        self.linear_k = nn.Dense(n_feat, n_feat)
        self.linear_v = nn.Dense(n_feat, n_feat)
        self.linear_out = nn.Dense(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        print(istrain)

    def construct(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.swapaxes(1, 2)  # (batch, head, time1, d_k)
        k = k.swapaxes(1, 2)  # (batch, head, time2, d_k)
        v = v.swapaxes(1, 2)  # (batch, head, time2, d_k)

        scores = ops.matmul(q, k.swapaxes(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(mindspore.Tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = ops.softmax(scores, axis=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = ops.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = ops.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = remove_head(p,x,istrain)
        x = x.swapaxes(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)
    

if __name__ == "__main__":
    batch_size = 4
    hidden_size = 16
    query_cnt = 8
    key_cnt = 8
    model = MultiHeadedAttention(4,hidden_size,0.1)
    query = ops.rand([batch_size, query_cnt, hidden_size])
    key = ops.rand([batch_size, key_cnt, hidden_size])
    value = ops.rand([batch_size, key_cnt, hidden_size])
    mask = ops.zeros([batch_size, query_cnt, key_cnt])
    print(model(query, key, value, mask))
