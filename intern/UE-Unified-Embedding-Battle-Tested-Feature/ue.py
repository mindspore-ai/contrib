import mindspore
from mindspore import nn, ops
import xxhash

class UnifiedEmbedding(nn.Cell):
    def __init__(self, emb_levels, emb_dim):
        super(UnifiedEmbedding, self).__init__()
        self.embedding = nn.Embedding(emb_levels, emb_dim)
        
    def construct(self, x, fnum):
        x_ = ops.zeros((x.shape[0], len(fnum)), mindspore.int64)
        for i in range(x.shape[0]):
            for j, h_seed in enumerate(fnum):
                x_[i, j] = xxhash.xxh32(x[i], h_seed).intdigest() % self.embedding.vocab_size
        return self.embedding(x_).reshape(x_.shape[0], -1)
        