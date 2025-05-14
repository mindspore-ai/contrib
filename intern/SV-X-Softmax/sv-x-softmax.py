import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, Parameter


class FC(nn.Cell):
    def __init__(
        self,
        fc_type='MV-AM',
        margin=0.35,
        t=0.2,
        scale=32,
        embedding_size=512,
        num_class=72690,
        easy_margin=True
    ):
        super(FC, self).__init__()
        self.weight = Parameter(
            Tensor(
                np.random.uniform(
                    -1, 1, (embedding_size, num_class)
                ).astype(np.float32)
            ),
            name='weight'
        )
        # L2 normalization for weights
        self.weight.set_data(
            self._renorm(self.weight.data, 2, 1, 1e-5) * 1e5
        )
        self.margin = margin
        self.t = t
        self.easy_margin = easy_margin
        self.scale = scale
        self.fc_type = fc_type
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.iter = 0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
        self.gather = ops.GatherD()
        self.scatter = ops.TensorScatterUpdate()
        self.arange = ops.Range()
        self.sqrt = ops.Sqrt()
        self.acos = ops.ACos()
        self.floor = ops.Floor()
        self.pow = ops.Pow()
        self.where = ops.Select()
        self.clamp = ops.clip_by_value

    def _renorm(self, x, p, dim, maxnorm):
        # Only supports p=2
        norm = ops.Sqrt()(
            ops.ReduceSum(keep_dims=True)(
                ops.Pow()(x, 2), dim
            )
        )
        desired = mnp.clip(norm, 0, maxnorm)
        x = x * (desired / (1e-8 + norm))
        return x

    def construct(self, x, label):
        # L2 normalization for input features
        kernel_norm = ops.L2Normalize(axis=0)(self.weight)
        cos_theta = ops.MatMul()(x, kernel_norm)
        cos_theta = self.clamp(cos_theta, -1, 1)
        batch_size = label.shape[0]
        arange_idx = mnp.arange(batch_size)
        gt = cos_theta[arange_idx, label].reshape((-1, 1))

        if self.fc_type == 'FC':
            final_gt = gt
        elif self.fc_type == 'SphereFace':
            self.iter += 1
            self.cur_lambda = max(
                self.lambda_min,
                self.base * (1 + self.alpha * self.iter) ** (-1 * self.power)
            )
            cos_theta_m = self.margin_formula[int(self.margin)](gt)
            theta = self.acos(gt)
            k = self.floor((self.margin * theta) / math.pi)
            phi_theta = ((-1.0) ** k) * cos_theta_m - 2 * k
            final_gt = (self.cur_lambda * gt + phi_theta)/(1 + self.cur_lambda)
        elif self.fc_type == 'AM':
            if self.easy_margin:
                final_gt = mnp.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin
        elif self.fc_type == 'Arc':
            sin_theta = self.sqrt(1.0 - mnp.power(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m
            if self.easy_margin:
                final_gt = mnp.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
        elif self.fc_type == 'MV-AM':
            mask = cos_theta > (gt - self.margin)
            hard_vector = cos_theta[mask]
            cos_theta = cos_theta.copy()
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t
            if self.easy_margin:
                final_gt = mnp.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin
        elif self.fc_type == 'MV-Arc':
            sin_theta = self.sqrt(1.0 - mnp.power(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta = cos_theta.copy()
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t
            if self.easy_margin:
                final_gt = mnp.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
        else:
            raise Exception('Unknown fc type!')

        # Scatter update for the ground truth positions
        cos_theta = cos_theta.copy()
        cos_theta[arange_idx, label] = final_gt.flatten()
        cos_theta = cos_theta * self.scale
        return cos_theta


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    mindspore.set_seed(42)

    batch_size = 4
    embedding_size = 512
    num_classes = 10

    # Create random input features and labels
    x = Tensor(
        np.random.randn(batch_size, embedding_size).astype(np.float32)
    )
    x = ops.L2Normalize(axis=1)(x)
    labels = Tensor(
        np.random.randint(0, num_classes, (batch_size,)),
        mindspore.int32
    )

    # Initialize FC layer
    fc_layer = FC(
        fc_type='MV-AM',
        margin=0.35,
        t=0.2,
        scale=32,
        embedding_size=embedding_size,
        num_class=num_classes,
        easy_margin=True
    )

    # Forward pass
    output = fc_layer(x, labels)

    # Print results
    print("Input feature shape:", x.shape)
    print("Labels:", labels)
    print("Output shape:", output.shape)
    print("Output sample:\n", output)
