from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore import Parameter

class LowerBound(nn.Cell):
    def __init__(self, bound):
        super(LowerBound, self).__init__()
        self.bound = Tensor(bound, mstype.float32)
        self.maximum = ops.Maximum()
        self.ones = ops.OnesLike()
        self.cast = ops.Cast()

    def construct(self, inputs):
        # 前向传播：计算 inputs 和 bound 的逐元素最大值
        bound_tensor = self.ones(inputs) * self.bound  # 创建和 inputs 相同形状的 bound 张量
        outputs = self.maximum(inputs, bound_tensor)
        return outputs

    def bprop(self, inputs, output, grad_output):
        # 反向传播
        bound_tensor = self.ones(inputs) * self.bound  # 创建和 inputs 相同形状的 bound 张量
        dtype = grad_output.dtype

        # 根据条件判断哪些位置可以通过梯度
        pass_through_1 = ops.GreaterEqual()(inputs, bound_tensor)
        pass_through_2 = ops.Less()(grad_output, Tensor(0, dtype))
        
        # 将条件组合
        pass_through = ops.LogicalOr()(pass_through_1, pass_through_2)

        # 转换 pass_through 的数据类型以匹配 grad_output，然后应用掩码
        grad_input = grad_output * self.cast(pass_through, dtype)
        return grad_input, None

class GDN(nn.Cell):
    """Generalized divisive normalization layer for MindSpore."""
  
    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = Tensor([reparam_offset], dtype=mstype.float32)

        self.build(ch)
  
    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = ops.Sqrt()(ops.Ones()((ch,), mstype.float32) + self.pedestal)
        self.beta = Parameter(beta, name="beta")

        # Create gamma param
        eye = ops.Eye()(ch, ch, mstype.float32)
        g = self.gamma_init * eye + self.pedestal
        gamma = ops.Sqrt()(g)
        self.gamma = Parameter(gamma, name="gamma")
        
        self.beta_lower_bound = LowerBound(self.beta_bound)
        self.gamma_lower_bound = LowerBound(self.gamma_bound)

    def construct(self, inputs):
        unfold = False
        if inputs.ndim == 5:
            unfold = True
            bs, ch, d, w, h = inputs.shape
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.shape

        # Beta bound and reparam
        # beta = LowerBound()(self.beta, self.beta_bound)
        beta = self.beta_lower_bound(self.beta)
        beta = beta ** 2 - self.pedestal 

        # Gamma bound and reparam
        # gamma = LowerBound()(self.gamma, self.gamma_bound)
        gamma = self.gamma_lower_bound(self.gamma)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = F.conv2d(inputs ** 2, gamma, beta)
        norm_ = ops.Sqrt()(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs
