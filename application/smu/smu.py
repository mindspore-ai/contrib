import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


class SMU(nn.Cell):
    '''
    Implementation of SMU activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    Examples:
        >>> smu = SMU()
        >>> x = ms.Tensor([0.6,-0.3])
        >>> x = smu(x)
    '''
    def __init__(self, alpha = 0.01, mu = 2.5):
        '''
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        '''
        super(SMU,self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = ms.Parameter(ms.tensor(mu)) 
        
    def construct(self, x):
        return ((1+self.alpha)*x + (1-self.alpha)*x*ops.erf(self.mu*(1-self.alpha)*x))/2
        
        
class SMU1(nn.Cell):
    '''
    Implementation of SMU-1 activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    Examples:
        >>> smu1 = SMU1()
        >>> x = ms.Tensor([0.6,-0.3])
        >>> x = smu1(x)
    '''
    def __init__(self, alpha = 0.01, mu = 4.332461424154261e-9):
        '''
        Initialization.
        INPUT:
            - alpha: hyper parameter
            aplha is initialized with zero value by default
        '''
        super(SMU1,self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = ms.Parameter(ms.tensor(mu)) 
        
    def construct(self, x):
        return ((1+self.alpha)*x+ops.sqrt(ops.square(x-self.alpha*x)+ops.square(self.mu)))/2
        
        
def test_SMU(x):
    smu_activation = SMU()
    print(smu_activation(x))
    
def test_SMU1(x):
    smu1_activation=SMU1()
    print(smu1_activation(x))

def test():
    x = ms.Tensor([0.6,-0.3])
    test_SMU(x)
    test_SMU1(x)

if __name__ == '__main__':
    test()
