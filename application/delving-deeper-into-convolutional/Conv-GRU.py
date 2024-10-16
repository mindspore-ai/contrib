
#####################################################
# Modified from:                                    #
# Nicolo Savioli, 2021 -- Conv-GRU mindspore v 1.0  #
#####################################################

import mindspore
from mindspore import nn, ops

class ConvGRUCell(nn.Cell):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, 3,\
                                     pad_mode="pad", padding=self.kernel_size // 2)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, 3,\
                                     pad_mode="pad", padding=self.kernel_size // 2) 
        dtype            = mindspore.Tensor
    
    def construct(self, input, hidden):
        if hidden is None:
            size_h   = [input.shape[0],self.hidden_size] + list(input.shape[2:])
            hidden   = ops.zeros(size_h)
        c1           = self.ConvGates(ops.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = ops.sigmoid(rt)
        update_gate  = ops.sigmoid(ut)
        gated_hidden = ops.mul(reset_gate,hidden)
        p1           = self.Conv_ct(ops.cat((input,gated_hidden),1))
        ct           = ops.tanh(p1)
        next_h       = ops.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h
 
def test(num_seqs, channels_img,\
         size_image, max_epoch, model): 
    input_image   = ops.rand(num_seqs,1,channels_img,size_image,size_image)
    target_image  = ops.rand(num_seqs,1,channels_img,size_image,size_image)
    # Create Autograd Variables
    input_gru     = input_image
    target_gru    = target_image
    # Create a MSE criterion
    MSE_criterion = nn.MSELoss()
    err           = 0
    h_next        = None
    for e in range(max_epoch):
        for time in range(num_seqs):
            h_next = model(input_gru[time], h_next)
            err   += MSE_criterion(h_next[0], target_gru[time])            
            print("... Error: "+ str(ops.stop_gradient(err).asnumpy()))
    
def main():
    num_seqs     = 10
    hidden_size  = 3
    channels_img = 3 
    size_image   = 256 
    max_epoch    = 100
    kernel_size  = 3
    print('Init Conv GRUs model:')
    model = ConvGRUCell(channels_img,hidden_size,kernel_size)
    print(repr(model))
    test(num_seqs, channels_img, size_image,\
         max_epoch, model)

if __name__ == '__main__':
    main()