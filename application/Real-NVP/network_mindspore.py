import numpy as np
import mindspore
from mindspore import nn, Tensor, Parameter
import mindspore.numpy as mnp


class RealNVP(nn.Cell):
    def __init__(self):
        super(RealNVP, self).__init__()
        self.num_scales = 2
        
        self.coupling_1 = CouplingLayer("01", 2)
        self.coupling_2 = CouplingLayer("10", 2)
        self.coupling_3 = CouplingLayer("01", 2)
        self.coupling_4 = CouplingLayer("10", 2)
        self.coupling_5 = CouplingLayer("01", 2)
        self.coupling_6 = CouplingLayer("10", 2)
        
        self.layers = nn.SequentialCell([])       
        
    def construct(self, x, reverse=False):
        if not reverse:
            sum_log_det_jacobians = mnp.zeros((x.shape[0],), x.dtype)
            
            z, log_det_jacobians = self.coupling_1(x, reverse)
            sum_log_det_jacobians = sum_log_det_jacobians + log_det_jacobians
            
            z, log_det_jacobians = self.coupling_2(z, reverse)
            sum_log_det_jacobians = sum_log_det_jacobians + log_det_jacobians
            
            z, log_det_jacobians = self.coupling_3(z, reverse)
            sum_log_det_jacobians = sum_log_det_jacobians + log_det_jacobians
            
            z, log_det_jacobians = self.coupling_4(z, reverse)
            sum_log_det_jacobians = sum_log_det_jacobians + log_det_jacobians
            
            z, log_det_jacobians = self.coupling_5(z, reverse)
            sum_log_det_jacobians = sum_log_det_jacobians + log_det_jacobians
            
            z, log_det_jacobians = self.coupling_6(z, reverse)
            sum_log_det_jacobians = sum_log_det_jacobians + log_det_jacobians

            return z, sum_log_det_jacobians
        else:
            output = self.coupling_6(x, reverse)
            output = self.coupling_5(output, reverse)
            output = self.coupling_4(output, reverse)
            output = self.coupling_3(output, reverse)
            output = self.coupling_2(output, reverse)
            output = self.coupling_1(output, reverse)
            
            return output


class CouplingLayer(nn.Cell):
    def __init__(self, mask_type, input_channel):
        super(CouplingLayer, self).__init__()
        self.function_s_t = Function_s_t(input_channel)
        self.mask_type = mask_type
            
    def get_mask(self, num):
        if '01' in self.mask_type:
            mask = Tensor([[0.0, 1.0]])
        else:
            mask = Tensor([[1.0, 0.0]])
        return mask
        
    def construct(self, x, reverse=False):
        if not reverse:
            mask = self.get_mask(self.mask_type)
            x1 = x * mask
            s, t = self.function_s_t(x1, mask)
            y = x1 + ((-mask + 1.0) * (x * mnp.exp(s) + t))
            log_det_jacobian = mnp.sum(s, axis=1)
            return y, log_det_jacobian
        else:
            mask = self.get_mask(self.mask_type)
            x1 = x * mask
            s, t = self.function_s_t(x1, mask)
            y = x1 + (-mask + 1.0) * ((x - t) * mnp.exp(-s))
            return y


class Function_s_t(nn.Cell):
    def __init__(self, input_channel, num_blocks=1, channel=256):
        super(Function_s_t, self).__init__()
        self.input_channel = input_channel
        
        layers = [
            nn.Dense(input_channel, channel),
            nn.LeakyReLU(),
            nn.Dense(channel, channel),
            nn.LeakyReLU(),
            nn.Dense(channel, input_channel * 2)
        ]
        
        self.model = nn.SequentialCell(layers)
        self.w_scale = Parameter(Tensor(np.random.rand(1), mindspore.float32), name="w_scale")
    
    def construct(self, x, mask):
        x = self.model(x)
        s = x[:, :self.input_channel] * (-mask + 1)
        t = x[:, self.input_channel:] * (-mask + 1)
        
        s = nn.Tanh()(s)
        
        return s, t
