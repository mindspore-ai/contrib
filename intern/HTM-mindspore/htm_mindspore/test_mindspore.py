import mindspore
import mindspore.ops as ops
from htm_mindspore import HTMAttention
from htm_mindspore import HTMBlock
mindspore.set_context(device_target="GPU")
attn = HTMAttention(
    dim = 512,
    heads = 8,               # number of heads for within-memory attention
    dim_head = 64,           # dimension per head for within-memory attention
    topk_mems = 8,           # how many memory chunks to select for
    mem_chunk_size = 32,     # number of tokens in each memory chunk
    add_pos_enc = True       # whether to add positional encoding to the memories
)

queries = ops.randn(1, 128, 512)     # queries
memories = ops.randn(1, 20000, 512)  # memories, of any size
mask = ops.ones((1, 20000),dtype=mindspore.float32).bool()     # memory mask


attended_mindspore = attn(queries, memories, mask = mask) # (1, 128, 512)
print('attn_mindspore:',attended_mindspore)

block = HTMBlock(
    dim = 512,
    topk_mems = 8,
    mem_chunk_size = 32
)

queries = ops.randn(1, 128, 512)
memories = ops.randn(1, 20000, 512)
mask = ops.ones((1, 20000),dtype=mindspore.float32).bool()  

out = block(queries, memories, mask = mask) # (1, 128, 512)
print('out',out)