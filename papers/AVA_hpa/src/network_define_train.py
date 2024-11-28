import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import ParameterTuple
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


class WithLossCell(nn.Cell):
    """
        Wrap the network with loss function to compute loss.

        Args:
            backbone (Cell): The target network to wrap.
            loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.concat = P.Concat()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        logits = self._backbone(data)
        return self._loss_fn(logits, label)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


class TrainOneStepCell(nn.Cell):
    """
        Network training package class.

        Append an optimizer to the training network after that the construct function
        can be called to create the backward graph.

        Args:
            net_with_loss (Cell): The training network with loss.
            optimizer (Cell): Optimizer for updating the weights.
            sens (Number): The adjust parameter. Default value is 1.0.
            reduce_flag (bool): The reduce flag. Default value is False.
            mean (bool): Allreduce method. Default value is False.
            degree (int): Device number. Default value is None.
    """

    def __init__(self, net_with_loss, optimizer, sens=1.0, reduce_flag=False, mean=False, degree=None):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.net_with_loss = net_with_loss
        self.weights = ParameterTuple(net_with_loss.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=False)
        # self.sens = Tensor((np.ones((1,)) * sens).astype(np.float32))
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, data, label, nslice):
        weights = self.weights
        loss = self.net_with_loss(data, label)
        grads = self.grad(self.net_with_loss, weights)(data, label)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))