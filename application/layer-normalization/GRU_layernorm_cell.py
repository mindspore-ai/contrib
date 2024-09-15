import mindspore as ms
import numpy as np
from mindspore import nn,ops,Tensor
from mindspore.common.initializer import Uniform
import math

class LayerNormGRUCell(nn.Cell):
    def __init__(self, input_size, hidden_size, has_bias=True):
        super().__init__()

        self.ln_i2h = nn.LayerNorm((2*hidden_size,))
        self.ln_h2h = nn.LayerNorm((2*hidden_size,))
        self.ln_cell_1 = nn.LayerNorm((hidden_size,))
        self.ln_cell_2 = nn.LayerNorm((hidden_size,))

        self.i2h = nn.Dense(input_size, 2 * hidden_size, has_bias=has_bias)
        self.h2h = nn.Dense(hidden_size, 2 * hidden_size, has_bias=has_bias)
        self.h_hat_W = nn.Dense(input_size, hidden_size, has_bias=has_bias)
        self.h_hat_U = nn.Dense(hidden_size, hidden_size, has_bias=has_bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        uniform_init = Uniform(std)  
        for w in self.trainable_params():
            new_data_tensor = Tensor(ops.zeros(w.data.shape,ms.float32))
            w.set_data(new_data_tensor)  

    def construct(self, x, h):

        h = h
        h = h.view(h.shape[0], -1)
        x = x.view(x.shape[0], -1)

        # Dense mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = ops.sigmoid(preact)
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = ops.tanh(  h_hat_first_half + ops.mul(r_t,   h_hat_last_half ) )

        h_t = ops.mul( 1-z_t , h ) + ops.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.reshape( h_t.shape[0], -1)
        return h_t

if __name__ == '__main__':
    ms.set_seed(1)
   
    batch_size = 4
    i_size = 8
    h_size = 16
    
    model = LayerNormGRUCell(i_size,h_size,has_bias=True)
    
    x = ms.Tensor(np.random.rand(batch_size,i_size),ms.float32)
    y = ms.Tensor(np.random.rand(batch_size,h_size),ms.float32)
    
    h_t = model(x,y)
    
    print(h_t)