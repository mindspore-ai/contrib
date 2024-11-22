import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as op
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer


# 共轭梯度信任域搜索
class ConjugateGradientTrustRegion(nn.Optimizer):
    def __init__(self, params, learning_rate=0.01, trust_region_radius=0.1, eps=1e-6):
        """
        初始化共轭梯度信任域搜索优化器。

        Args:
            params (list): 需要优化的参数列表
            learning_rate (float, optional): 学习率，默认值为0.01。
            trust_region_radius (float, optional): 信任域半径，用于限制参数更新的范围，默认值为0.1。
            eps (float, optional): 一个极小值，用于避免数值计算中的除零等问题，默认值为1e-6。
        """
        super(ConjugateGradientTrustRegion, self).__init__(learning_rate, params, weight_decay=0.0)
        if trust_region_radius <= 0:
            raise ValueError("Trust region radius should be positive.")
        if eps <= 0:
            raise ValueError("Epsilon should be positive.")
        self.trust_region_radius = trust_region_radius
        self.eps = eps
        self.parameters = params
        self.memory = {}
        self.step_size = 1.0
        for p in self.parameters:
            self.memory[p.name] = {"grad": None, "search_direction": None}

    def construct(self, gradients):
        """
        执行一次优化器的更新步骤。

        Args:
            gradients (list): 对应参数的梯度列表，与参数列表顺序一致。
        """
        if len(gradients)!= len(self.parameters):
            raise ValueError("The number of gradients should match the number of parameters.")
        
        for p, grad in zip(self.parameters, gradients):
            if grad is None:
                continue
            search_direction = self.compute_search_direction(p, grad)
            step_size = self.compute_step_size(p, search_direction)
            new_param_value = p + step_size * search_direction
            op.assign(p, new_param_value)

    def compute_search_direction(self, param, grad):
        """
        计算搜索方向。

        Args:
            param (Parameter): 需要更新的参数。
            grad (Tensor): 参数对应的梯度。

        Returns:
            Tensor: 计算得到的搜索方向。
        """
        if self.memory[param.name]["search_direction"] is None:
            # 首次使用负梯度作为搜索方向
            search_direction = -grad
        else:
            prev_search_direction = self.memory[param.name]["search_direction"]
            prev_grad = self.memory[param.name]["grad"]
            beta = self.compute_beta(grad, prev_grad)
            search_direction = -grad + beta * prev_search_direction

        # 归一化搜索方向，添加eps防止范数为零的情况
        norm_value = op.norm(search_direction)
        if norm_value < self.eps:
            search_direction = search_direction / self.eps
        else:
            search_direction = search_direction / (norm_value + self.eps)
        self.memory[param.name]["search_direction"] = search_direction
        self.memory[param.name]["grad"] = grad
        return search_direction

    def compute_beta(self, grad, prev_grad):
        """
        使用Fletcher-Reeves公式计算beta。

        Args:
            grad (Tensor): 当前梯度。
            prev_grad (Tensor): 上一次的梯度。

        Returns:
            float: 计算得到的beta值。
        """
        numerator = op.sum(grad * (grad - prev_grad))

        denominator = op.sum(prev_grad * prev_grad)

        if denominator < self.eps:
            return 0.0  # 防止分母为零
        beta = numerator / denominator
        return beta

    def compute_step_size(self, param, search_direction):
        """
        计算步长。

        Args:
            param (Parameter): 需要更新的参数。
            search_direction (Tensor): 计算得到的搜索方向。

        Returns:
            float: 合适的步长值。
        """
        norm_value = op.norm(search_direction)
        if norm_value < self.eps:
            step_size = self.trust_region_radius / self.eps
        else:
            step_size = self.trust_region_radius / (norm_value + self.eps)
        self.step_size = step_size
        return step_size
# test
# net = nn.Dense(10, 10)
# params = net.trainable_params()
# # 创建优化器实例
# optimizer = ConjugateGradientTrustRegion(params)

# # 模拟梯度
# gradients = [op.rand_like(p) for p in params]

# # 执行一次优化器更新
# optimizer(gradients)