import mindspore.nn as nn

class Loss(nn.Cell):
    def __init__(self, prior):
        super(Loss, self).__init__()
        self.prior = prior
        
    def construct(self, z, sum_log_det_jacobians):
        log_p = self.prior.log_prob(z)
        return -(log_p + sum_log_det_jacobians).mean()
