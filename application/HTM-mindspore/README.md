This code is a mindspore implementation of Towards mental time travel: a hierarchical memory for reinforcement learning agents which is available at https://github.com/lucidrains/HTM-pytorch paperswidthcocde link is https://paperswithcode.com/paper/towards-mental-time-travel-a-hierarchical

<img src="./htm.png" width="500px"></img>

## Hierarchical Transformer Memory (HTM) - Mindspore

Implementation of <a href="https://arxiv.org/abs/2105.14039">Hierarchical Transformer Memory</a> (HTM) for Mindspore. This Deepmind paper proposes a simple method to allow transformers to attend to memories of the past efficiently. 

## Install

```bash
$ pip install htm-mindspore
```

## Usage

```python
import mindspore
import mindspore.ops as ops
from htm_mindspore import HTMAttention
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

If you want the entire HTM Block (which contains the layernorm for the input followed by a skip connection), just import `HTMBlock` instead

```python
import mindspore
import mindspore.ops as ops
from htm_mindspore import HTMAttention
mindspore.set_context(device_target="GPU")
from htm_mindspore import HTMBlock

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
```

## Citations

```bibtex
@misc{lampinen2021mental,
    title   = {Towards mental time travel: a hierarchical memory for reinforcement learning agents}, 
    author  = {Andrew Kyle Lampinen and Stephanie C. Y. Chan and Andrea Banino and Felix Hill},
    year    = {2021},
    eprint  = {2105.14039},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
