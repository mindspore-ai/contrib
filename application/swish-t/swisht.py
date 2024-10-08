import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Parameter


class SwishT(nn.Cell):
    def __init__(self, beta_init=1.0, alpha=0.1, requires_grad=True):
        super().__init__()
        self.beta = Parameter(ms.tensor([beta_init]), requires_grad=requires_grad)  
        self.alpha = alpha  # Could also be made learnable if desired

    def construct(self, x):
        return x * ops.sigmoid(self.beta * x) + self.alpha * ops.tanh(x)

class SwishT_A(nn.Cell):
    def __init__(self, alpha=0.1,):
        super().__init__()
        self.alpha = alpha

    def construct(self, x):
        return ops.sigmoid(x) * (x + 2 * self.alpha) - self.alpha

class SwishT_B(nn.Cell):
    def __init__(self, beta_init=1.0, alpha=0.1, requires_grad=True):
        super().__init__()
        self.beta = Parameter(ms.tensor([beta_init]), requires_grad=requires_grad)  
        self.alpha = alpha  

    def construct(self, x):
        return ops.sigmoid(self.beta * x) * (x + 2 * self.alpha) - self.alpha

class SwishT_C(nn.Cell):
    def __init__(self, beta_init=1.0, alpha=0.1, requires_grad=True):
        super().__init__()
        self.beta = Parameter(ms.tensor([beta_init]), requires_grad=requires_grad)  
        self.alpha = alpha  

    def construct(self, x):
        return ops.sigmoid(self.beta * x) * (x + 2 * self.alpha / self.beta) - self.alpha / self.beta
    

if __name__ == '__main__':
    x = ops.linspace(-3, 3, 50)
    
    for [act, name] in [[SwishT(), 'SwishT'], [SwishT_A(), 'SwishT_A'], [SwishT_B(), 'SwishT_B'], [SwishT_C(), 'SwishT_C']]:
        out = act(x)
        print(f'{name} output shape: {act(x).shape}') 
        print(out)
