import mindspore 
import mindspore.nn as ms_nn
import mindspore.ops as ops
import fisherform
import fishertrace
ms_net = ms_nn.SequentialCell(ms_nn.Dense(10,5),ms_nn.Tanh(),ms_nn.Dense(5,3),ms_nn.LogSoftmax(axis=1))
ms_x = ops.randn((1,10))
def forward(ms_x):
    return ms_net(ms_x).max()
grad_fn = mindspore.value_and_grad(forward,has_aux=False)
ms_out = ms_net(ms_x).max()
output, inputs_gradient = grad_fn(ms_x)
ms_v = []
for ms_par in ms_net.get_parameters():
    ms_v.append(ms_par.value().copy())
fisherform.numeric_fisher_form(ms_x, ms_net, ms_v)
fisherform.fisher_information_sensitivity(ms_x, ms_net, ms_v) 
fishertrace.compute_fisher_trace(ms_x, ms_net, ms_v)