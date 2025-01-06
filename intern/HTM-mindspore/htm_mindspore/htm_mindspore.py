from math import ceil
import mindspore
from mindspore import Tensor,Parameter
from mindspore import dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from einops import rearrange, repeat
import numpy as np
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_to_multiple(t, multiple, dim = -2, value = 0.):
    seq_len = t.shape[dim]
    pad_to_len = ceil(seq_len / multiple) * multiple
    remainder = pad_to_len - seq_len

    if remainder == 0:
        return t

    zeroes = (0, 0) * (-dim - 1)
    padded_t = ops.pad(t, (*zeroes, remainder, 0), value = value)
    # print('padded_t:',padded_t)
    return padded_t

# positional encoding

class SinusoidalPosition(nn.Cell):
    def __init__(
        self,
        dim,
        min_timescale = 2.,
        max_timescale = 1e4
    ):
        super().__init__()
        freqs = ops.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.inv_freqs = Parameter(inv_freqs, requires_grad=False)

    def construct(self, x):
        seq_len = x.shape[-2]
        seq = ops.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = Tensor(rearrange(seq.asnumpy(), 'n -> n ()'),mindspore.float32) * Tensor(rearrange((self.inv_freqs).asnumpy(), 'd -> () d'),mindspore.float32)
        pos_emb = ops.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), axis = -1)
        return pos_emb

# multi-head attention

class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Dense(dim, inner_dim, has_bias = False)
        self.to_kv = nn.Dense(dim, inner_dim * 2, has_bias = False)
        self.to_out = nn.Dense(inner_dim, dim)

    def construct(
        self,
        x,
        mems,
        mask = None
    ):
        h = self.heads
        q, k, v = self.to_q(x), *self.to_kv(mems).chunk(2, axis = -1)

        q, k, v = map(lambda t: Tensor(rearrange(t.asnumpy(), 'b ... (h d) -> (b h) ... d', h = h),mindspore.float32), (q, k, v))
        q = q * self.scale
        
        sim = ops.einsum('b m i d, b m i j d -> b m i j', q, k)
        if exists(mask):
            mask = Tensor(repeat(mask.asnumpy(), 'b ... -> (b h) ...', h = h),mindspore.float32)
            mask_value = -np.finfo(sim.asnumpy().dtype).max
            mask_value=Tensor(mask_value,dtype=mindspore.float32)
            sim = sim.masked_fill(~mask, mask_value)

        attn = ops.softmax(sim,axis = -1)

        out = ops.einsum('... i j, ... i j d -> ... i d', attn, v)
        out = Tensor(rearrange(out.asnumpy(), '(b h) ... d -> b ... (h d)', h = h),mindspore.float32)
        return self.to_out(out)

# main class

class HTMAttention(nn.Cell):
    def __init__(
        self,
        dim,
        heads,
        topk_mems = 2,
        mem_chunk_size = 32,
        dim_head = 64,
        add_pos_enc = True,
        eps = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_summary_queries = nn.Dense(dim, dim)
        self.to_summary_keys = nn.Dense(dim, dim)

        self.attn = Attention(dim = dim, heads = heads, dim_head = dim_head)

        self.topk_mems = topk_mems
        self.mem_chunk_size = mem_chunk_size
        self.pos_emb = SinusoidalPosition(dim = dim) if add_pos_enc else None

    def construct(
        self,
        queries,
        memories,
        mask = None,
        chunk_attn_mask = None
    ):
        dim, query_len, mem_chunk_size, topk_mems, scale, eps = self.dim, queries.shape[1], self.mem_chunk_size, self.topk_mems, self.scale, self.eps

        # pad memories, and the memory mask, if needed
        # and then divide into chunks

        memories = pad_to_multiple(memories, mem_chunk_size, dim = -2, value = 0.)
        rearranged_memories_np = rearrange(memories.asnumpy(), 'b (n c) d -> b n c d', c = mem_chunk_size)
        memories = Tensor(rearranged_memories_np, mindspore.float32)
       
        if exists(mask):
            mask = pad_to_multiple(mask, mem_chunk_size, dim = -1, value = False)
            rearranged_mask_np = rearrange(mask.asnumpy(), 'b (n c) -> b n c', c = mem_chunk_size)
            mask = Tensor(rearranged_mask_np,mindspore.float32)
       
        # summarize memories through mean-pool, accounting for mask

        if exists(mask):
            rearranged_mean_mask_np = rearrange(mask.asnumpy(), '... -> ... ()')
            mean_mask = Tensor(rearranged_mean_mask_np , mindspore.float32)
            memories = memories.masked_fill(~mean_mask, 0.)
            numer = memories.sum(axis = 2)
            denom = mean_mask.sum(axis = 2)
            summarized_memories = numer / (denom + eps)
        else:
            summarized_memories = memories.mean(axis = 2)

        # derive queries and summarized memory keys

        summary_queries = self.to_summary_queries(queries)
        summary_keys =Parameter(self.to_summary_keys(summarized_memories),requires_grad=False)

        # do a single head attention over summary keys
        sim = ops.einsum('b i d, b j d -> b i j', summary_queries, summary_keys) * scale
        mask_value = -np.finfo(sim.asnumpy().dtype).max
        mask_value=Tensor(mask_value,dtype=mindspore.float32)
        
        if exists(mask):
            bool_mask =Tensor(mask,dtype=mindspore.bool_)
            chunk_mask = Tensor.any(bool_mask,axis = 2)
            
            rearranged_chunk_mask_np = rearrange(chunk_mask.asnumpy(), 'b j -> b () j')
            chunk_mask =Tensor(rearranged_chunk_mask_np,mindspore.float32)
            sim = sim.masked_fill(~chunk_mask, mask_value)

        if exists(chunk_attn_mask):
            sim = sim.masked_fill(~chunk_attn_mask, mask_value)

        topk_logits, topk_indices = sim.topk(k = topk_mems, dim = -1)
        topk_logits=Tensor(topk_logits,mindspore.float32)
        weights = ops.softmax(topk_logits,axis=-1)
       
        # ready queries for in-memory attention
        
        repeated_queries_np = repeat(queries.asnumpy(), 'b n d -> b k n d', k = topk_mems)
        queries = Tensor(repeated_queries_np,mindspore.float32)
        
        # select the topk memories

        repeated_memories_np = repeat(memories.asnumpy(), 'b m j d -> b m i j d', i = query_len)
        memories = Tensor(repeated_memories_np,mindspore.float32)
        repeated_mem_topk_indices_np = repeat(topk_indices.asnumpy(), 'b i m -> b m i j d', j = mem_chunk_size, d = dim)
        mem_topk_indices=Tensor(repeated_mem_topk_indices_np,mindspore.int32)
        selected_memories = memories.gather_elements(1, mem_topk_indices)

        # positional encoding

        if exists(self.pos_emb):
            pos_emb = self.pos_emb(memories)
            selected_memories = selected_memories + Tensor(rearrange(pos_emb.asnumpy(), 'n d -> () () () n d'),mindspore.float32)

        # select the mask

        selected_mask = None
        if exists(mask):
            mask = Tensor(repeat(mask.asnumpy(), 'b m j -> b m i j', i = query_len),mindspore.float32)
            mask_topk_indices = Tensor(repeat(topk_indices.asnumpy(), 'b i m -> b m i j', j = mem_chunk_size),mindspore.int32)
            selected_mask = mask.gather_elements(1, mask_topk_indices)

        # now do in-memory attention
        
        copy_selected_memories=selected_memories.copy()
        copy_selected_memories.requires_grad = False
        
        within_mem_output = self.attn(
            queries,
            copy_selected_memories,
            mask = selected_mask
        )

        # weight the in-memory attention outputs

        weighted_output = within_mem_output * Tensor(rearrange(weights.asnumpy(), 'b i m -> b m i ()'),mindspore.float32)
        output = Tensor.sum(weighted_output , axis = 1)
        return output

# HTM Block

class HTMBlock(nn.Cell):
    def __init__(self, dim, heads=8, dim_head=64, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm([dim])
        self.attn = HTMAttention(dim = dim,heads=heads,dim_head=dim_head,**kwargs)
    def construct(
        self,
        queries,
        memories,
        **kwargs
    ):
        queries = self.norm(queries)
        out = self.attn(queries, memories, **kwargs) + queries
        return out