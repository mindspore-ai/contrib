import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Uniform
import numpy as np


# Code modified originally from https://vsitzmann.github.io/siren/
class SineLayer(nn.Cell):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, has_bias = True,
                 is_first = False, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        if self.is_first:
            self.linear = nn.Dense(in_features, out_features, 
                weight_init = initializer(Uniform(1 / self.in_features), [out_features, in_features]), 
                has_bias = has_bias)
        else:
            self.linear = nn.Dense(in_features, out_features, 
                weight_init = initializer(Uniform(np.sqrt(6 / self.in_features) / self.omega_0), [out_features, in_features]), 
                has_bias = has_bias)
     
    def construct(self, input):
        return ops.sin(self.omega_0 * self.linear(input))


class Siren(nn.Cell):
    def __init__(self, features, outermost_linear = False, first_omega_0 = 30, 
                 hidden_omega_0 = 30.0):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(features[0], features[1], 
                                  is_first=True, omega_0 = first_omega_0))

        for i in range(1, len(features) - 2):
            self.net.append(SineLayer(features[i], features[i + 1], 
                                      is_first = False, omega_0 = hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Dense(features[-2], features[-1],
                weight_init = initializer(Uniform(np.sqrt(6 / features[-2]) / hidden_omega_0), [features[-1], features[-2]]))
            
                
            self.net.append(final_linear)
            #adding Tanh to original implementation to rescale in [-1, 1]
            self.net.append(nn.Tanh())
        else:
            self.net.append(SineLayer(features[-2], features[-1], 
                                      is_first = False, omega_0 = hidden_omega_0))
        
        self.net = nn.SequentialCell(*self.net)
    
    def construct(self, coords):
        output = self.net(coords)
        return output        
