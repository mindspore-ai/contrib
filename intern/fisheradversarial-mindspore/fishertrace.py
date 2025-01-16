
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

def compute_fisher_trace(input, net, parameter_generator=None):
    '''
    Computes the trace of the Fisher matrix for a Categorical model. Expecting **linear** output.

    :param input: A (single) input suitable for net
    :param net: A nn.Cell object with linear output
    :param parameter_generator: If None, all parameters will be considered (net.trainable_params() is taken)
    :return: The Fisher information for the input
    '''
    print('compute_fisher_trace')
    input = Tensor(input)
    test_output = net(input)
    assert test_output.shape[0] == 1 and len(test_output.shape) == 2
    output_dim = test_output.shape[1]
    
    fisher_trace = 0

    if parameter_generator is None:
        parameter_generator = net.trainable_params()
    
    softmax = ops.Softmax(axis=1)
    
    for j in range(output_dim):
        log_softmax_output = ops.LogSoftmax(axis=1)(net(input))
        grad_op = ms.ops.GradOperation(get_all=True)
        log_softmax_grad = grad_op(net)(input)[0][0, j]
            
        softmax_output = softmax(net(input))
        grad_op = ms.ops.GradOperation(get_all=True)
        softmax_grad = grad_op(net)(input)[0][0, j]

        fisher_trace += (log_softmax_grad * softmax_grad).sum()
    print('fisher_trace:',fisher_trace)
    return fisher_trace
