import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class APLLoss(nn.Cell):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def construct(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = ops.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (ops.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * ops.pow(1 - xs_pos.clamp(min=self.eps), 2) )
        los_neg = (1 - y) * (ops.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) )
        loss = los_pos + los_neg
        ttfa:mindspore.Tensor = x_sigmoid
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # if self.disable_torch_grad_focal_loss:
            #     torch.set_grad_enabled(False)

            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = ops.pow(1 - pt, one_sided_gamma)
            # if self.disable_torch_grad_focal_loss:
            #     torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
    

if __name__ == "__main__":
    losser = APLLoss()
    logits = mindspore.Tensor([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]],dtype=mindspore.float32)
    expected = mindspore.Tensor([[2],[1],[0],[3]],dtype=mindspore.float32)
    print(losser(logits,expected))
