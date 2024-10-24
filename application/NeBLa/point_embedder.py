import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


ms.set_context(device_target='Ascend', device_id=0)

# input points are embedded through this embedder and concatenated with encoded image and propagated to MLP model.

class Embedder(nn.Cell):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** np.linspace(0., max_freq, N_freqs)
            freq_bands = ms.Tensor(freq_bands, dtype=ms.float32)
        else:
            freq_bands = np.linspace(2.0 ** 0., 2.0 ** max_freq, N_freqs)
            freq_bands = ms.Tensor(freq_bands, dtype=ms.float32)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    def construct(self, inputs):
        return ops.Concat(axis=-1)([fn(inputs) for fn in self.embed_fns])


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [ops.Sin(), ops.Cos()],
    }

    return Embedder(**embed_kwargs), embed_kwargs['input_dims'] * (embed_kwargs['num_freqs'] * 2)


def main():
    
    multires = 10
    batch_size = 4
    num_samples = 10 

    embedder, out_dim = get_embedder(multires)
    inputs = ops.randn((batch_size, num_samples, 3))
    embedded_inputs = embedder(inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"Embedded shape: {embedded_inputs.shape}")
    

if __name__ == "__main__":
    main()
