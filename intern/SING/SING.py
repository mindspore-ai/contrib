import mindspore
from mindspore import nn, ops, Tensor

class SING(nn.Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, clip_value=1.0,
                la_active=True,la_mergetime=5,la_alpha=0.5):
        super().__init__(learning_rate, params, weight_decay)
        self.beta1 = Tensor(beta1, mindspore.float32)
        self.beta2 = Tensor(beta2, mindspore.float32)
        self.eps = Tensor(eps, mindspore.float32)
        self.clip_value = clip_value
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.step_size = self.parameters.clone(prefix="adam_step", init='zeros')
        self.opt = ops.ApplyMomentum()
        self.la_active=la_active
        if self.la_active:
            self.lookahead_params=self.parameters.clone(prefix="la_param",init='same')
            self.la_alpha=la_alpha
            self.la_mergetime=la_mergetime

    def construct(self, gradients):
        lr = self.get_lr()
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        
        updated_parameters = []
        for i, param in enumerate(self.parameters):
            gradient = gradients[i]
            # Gradient clipping
            gradient = ops.clip_by_value(gradient, -self.clip_value, self.clip_value)

            #normal Adam
            m = self.moments1[i]
            v = self.moments2[i]
            step = self.step_size[i]

            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * gradient
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * ops.square(gradient)
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - ops.pow(self.beta1, step + 1))
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - ops.pow(self.beta2, step + 1))
            # Update parameters
            update = lr * m_hat / (ops.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                update += self.weight_decay * param
            updated_param = param - update
            updated_parameters.append(updated_param)

            # Update moments and step size
            ops.assign(self.moments1[i], m)
            ops.assign(self.moments2[i], v)
            ops.assign(self.step_size[i], step + 1)
            
            if not self.la_active:
                ops.assign(self.parameters[i],updated_param)
            #use lookahead
            else:
                la_param=self.lookahead_params[i]
                updated_param.mul(self.la_alpha).add(la_param.mul(1-self.la_alpha))
                ops.assign(self.lookahead_params[i],updated_param)
                ops.assign(self.parameters[i],updated_param)

        return updated_parameters