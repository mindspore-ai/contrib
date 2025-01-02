import warnings
import numpy.core.fromnumeric

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import mindspore as ms
from mindspore import context
from mindspore_convgru import ConvGRU

def main():
    # Set context
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.set_seed(42)
    
    # Model parameters
    input_size = 8
    hidden_sizes = [32, 64, 16]
    kernel_sizes = [3, 5, 3]
    n_layers = 3
    
    # Create model
    model = ConvGRU(input_size=input_size, 
                    hidden_sizes=hidden_sizes,
                    kernel_sizes=kernel_sizes, 
                    n_layers=n_layers)
    
    # Generate sample input (batch_size=2, channels=8, height=64, width=64)
    batch_size = 2
    height = 64
    width = 64
    x = ms.numpy.randn(batch_size, input_size, height, width)
    
    # Forward pass
    output = model(x)
    
    # Print information about the output
    print(f"Number of output feature maps: {len(output)}")
    for i, out in enumerate(output):
        print(f"Layer {i} output shape: {out.shape}")

if __name__ == "__main__":
    main()
