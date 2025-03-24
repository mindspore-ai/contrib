import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.common.initializer import Normal


# Define the aw_method class
class aw_method():
    def __init__(self, alpha1=0.5, alpha2=0.75, delta=0.05, epsilon=0.05, normalized_aw=True):
        assert alpha1 < alpha2
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._delta = delta
        self._epsilon = epsilon
        self._normalized_aw = normalized_aw

    def aw_loss(self, Dis_opt, Dis_Net, real_data, fake_data):
        # 计算 real and fake validity
        real_validity = Dis_Net(real_data)
        fake_validity = Dis_Net(fake_data)

        # 使用正确的标签计算真实和虚假损失
        loss_fn_real = nn.BCEWithLogitsLoss()
        loss_fn_fake = nn.BCEWithLogitsLoss()
        Dloss_real = loss_fn_real(real_validity, ops.ones_like(real_validity))
        Dloss_fake = loss_fn_fake(fake_validity, ops.zeros_like(fake_validity))

        # 计算真实数据损失和梯度的函数
        def real_loss(data):
            validity = Dis_Net(data)
            loss = loss_fn_real(validity, ops.ones_like(validity))
            return loss

        grad_real_fn = ms.value_and_grad(real_loss, grad_position=None, weights=Dis_Net.trainable_params())
        real_loss_val, grad_real_tensor = grad_real_fn(real_data)

        # 展平并连接真实数据梯度
        grad_real_list = ops.concat([grad.reshape(-1) for grad in grad_real_tensor], axis=0)
        rdotr = ops.matmul(grad_real_list, grad_real_list) + 1e-4
        rdotr_value = rdotr.asnumpy().item()  # 转换为Python标量
        r_norm = np.sqrt(rdotr_value)

        # 计算假数据损失和梯度的函数
        def fake_loss(data):
            validity = Dis_Net(data)
            loss = loss_fn_fake(validity, ops.zeros_like(validity))
            return loss

        grad_fake_fn = ms.value_and_grad(fake_loss, grad_position=None, weights=Dis_Net.trainable_params())
        fake_loss_val, grad_fake_tensor = grad_fake_fn(fake_data)

        # 展平并连接假数据梯度
        grad_fake_list = ops.concat([grad.reshape(-1) for grad in grad_fake_tensor], axis=0)
        fdotf = ops.matmul(grad_fake_list, grad_fake_list) + 1e-4
        fdotf_value = fdotf.asnumpy().item()  # 转换为Python标量
        f_norm = np.sqrt(fdotf_value)

        # 计算真实梯度和假梯度之间的点积
        rdotf = ops.matmul(grad_real_list, grad_fake_list).asnumpy().item()

        fdotr = rdotf


        rs = ops.sigmoid(real_validity).mean()
        fs = ops.sigmoid(fake_validity).mean()

        # 根据条件确定权重
        if self._normalized_aw:
            if rs < self._alpha1 or rs < (fs - self._delta):
                if rdotf <= 0:
                    w_r = (1 / r_norm) + self._epsilon
                    w_f = (-fdotr / (fdotf * r_norm)) + self._epsilon
                else:
                    w_r = (1 / r_norm) + self._epsilon
                    w_f = self._epsilon
            elif rs > self._alpha2 and rs > (fs - self._delta):
                if rdotf <= 0:
                    w_r = (-rdotf / (rdotr * f_norm)) + self._epsilon
                    w_f = (1 / f_norm) + self._epsilon
                else:
                    w_r = self._epsilon
                    w_f = (1 / f_norm) + self._epsilon
            else:
                w_r = (1 / r_norm) + self._epsilon
                w_f = (1 / f_norm) + self._epsilon
        else:
            if rs < self._alpha1 or rs < (fs - self._delta):
                if rdotf <= 0:
                    w_r = 1 + self._epsilon
                    w_f = (-fdotr / fdotf) + self._epsilon
                else:
                    w_r = 1 + self._epsilon
                    w_f = self._epsilon
            elif rs > self._alpha2 and rs > (fs - self._delta):
                if rdotf <= 0:
                    w_r = (-rdotf / rdotr) + self._epsilon
                    w_f = 1 + self._epsilon
                else:
                    w_r = self._epsilon
                    w_f = 1 + self._epsilon
            else:
                w_r = 1 + self._epsilon
                w_f = 1 + self._epsilon

        # 计算 aw_loss
        aw_loss = w_r * Dloss_real + w_f * Dloss_fake

        # 使用计算出的权重更新梯度
        for index, param in enumerate(Dis_Net.trainable_params()):
            param.grad = w_r * grad_real_tensor[index] + w_f * grad_fake_tensor[index]

        return aw_loss


# 定义一个 discriminator network
class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.SequentialCell([
            nn.Dense(2, 128, weight_init=Normal(0.02)),
            nn.ReLU(),
            nn.Dense(128, 1, weight_init=Normal(0.02))
        ])

    def construct(self, x):
        return self.model(x)


def main():
    # 设置使用CPU
    ms.set_context(device_target="CPU")

    # 初始化discriminator network
    Dis_Net = Discriminator()

    # 初始化optimizer
    Dis_opt = nn.Adam(Dis_Net.trainable_params(), learning_rate=0.001)

    # 定义aw_method
    aw = aw_method()

    # 创建测试数据
    real_data = Tensor(np.random.randn(16, 2), ms.float32)
    fake_data = Tensor(np.random.randn(16, 2), ms.float32)

    # 计算自适应加权损失和更新梯度
    aw_loss_value = aw.aw_loss(Dis_opt, Dis_Net, real_data, fake_data)

    # print(type(aw_loss_value))
    # print("AW Loss:", aw_loss_value) # mindspore2.5.0结果为标量
    print("AW Loss:", aw_loss_value.asnumpy())  # mindspore2.3.0结果为张量


if __name__ == "__main__":
    main()