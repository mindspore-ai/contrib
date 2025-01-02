import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Orthogonal, Zero

class ConvGRUCell(nn.Cell):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        weight_init = Orthogonal()
        bias_init = Zero()
        
        # Create convolution layers
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 
                                  kernel_size=kernel_size, pad_mode='pad', 
                                  padding=padding, weight_init=weight_init, 
                                  bias_init=bias_init)
        
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 
                                   kernel_size=kernel_size, pad_mode='pad', 
                                   padding=padding, weight_init=weight_init, 
                                   bias_init=bias_init)
        
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 
                                kernel_size=kernel_size, pad_mode='pad', 
                                padding=padding, weight_init=weight_init, 
                                bias_init=bias_init)
        
        # Define operations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.concat = ops.Concat(axis=1)

    def construct(self, input_tensor, prev_state):
        # Get tensors from previous state
        if prev_state is None:
            batch_size = input_tensor.shape[0]
            spatial_size = input_tensor.shape[2:]
            prev_state = ops.zeros((batch_size, self.hidden_size, *spatial_size), 
                                 input_tensor.dtype)
        
        # Concatenate input and previous state
        stacked_inputs = self.concat((input_tensor, prev_state))
        
        # Apply gates
        reset = self.sigmoid(self.reset_gate(stacked_inputs))
        update = self.sigmoid(self.update_gate(stacked_inputs))
        
        # Compute new state
        reset_hidden = self.concat((input_tensor, reset * prev_state))
        out = self.tanh(self.out_gate(reset_hidden))
        new_state = (1 - update) * prev_state + update * out
        
        return new_state


class ConvGRU(nn.Cell):
    """
    Generates a multi-layer convolutional GRU
    """
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        """
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained ConvGRUCell.
        """
        super().__init__()
        
        self.input_size = input_size
        
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
            
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes
        
        self.n_layers = n_layers
        
        # Create cells as individual attributes
        for i in range(self.n_layers):
            input_dim = self.input_size if i == 0 else self.hidden_sizes[i-1]
            setattr(self, f'cell_{i}', 
                   ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i]))

    def construct(self, x, hidden=None):
        """
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        
        Returns
        -------
        upd_hidden : list of 4D hidden representation. (batch, channels, height, width).
        """
        if hidden is None:
            hidden = [None] * self.n_layers
        elif not isinstance(hidden, (list, tuple)):
            hidden = [hidden]  # Convert single tensor to list
        
        input_ = x
        upd_hidden = []
        
        # Process through each layer
        for layer_idx in range(self.n_layers):
            cell = getattr(self, f'cell_{layer_idx}')
            cell_hidden = hidden[layer_idx] if layer_idx < len(hidden) else None
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            input_ = upd_cell_hidden
        
        return upd_hidden
