"""Mindspore utils module."""
from copy import deepcopy
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore import  ParameterTuple, Tensor

class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2):
        output = self._backbone(data)
        return self._loss_fn(output, label1, label2)

class GradWrap(nn.Cell):
    """ GradWrap definition """
    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label, edge1, edge2, weight):
        weights = self.weights
        return C.GradOperation(get_by_list=True)(self.network, weights)(x, label, edge1, edge2, weight)

def gen_checkpoints_list(params):
    shadow = {}
    for _, param in params:
        shadow[param.name] = deepcopy(param.data.asnumpy())
    output = [{"name": k, "data": Tensor(v)} for k, v in shadow.items()]
    return output
