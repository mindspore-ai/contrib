import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE)  # 保持动态图模式


# 修正后的clip函数（关键修正点）
def clip(x, x_, eps):
    # 修正1：使用reduce_max替代maximum
    stack_max = ops.stack([x - eps, x_], axis=0)
    lower_clip = ops.maximum(ops.reduce_max(stack_max, axis=0), ops.zeros_like(x))

    # 修正2：使用reduce_min替代minimum
    stack_min = ops.stack([x + eps, lower_clip], axis=0)
    return ops.minimum(ops.reduce_min(stack_min, axis=0), ops.ones_like(x))


def train_adv_examples(model, loss_fct, adv_examples, adv_targets,
                       epochs=1, alpha=8 / 255, clip_eps=8 / 255,
                       do_clip=True, minimize=False):
    model.set_train(False)
    grad_fn = mindspore.value_and_grad(lambda x: loss_fct(model(x), adv_targets), grad_position=0)

    for _ in range(epochs):
        loss, adv_grad = grad_fn(adv_examples)
        direction = -1 if minimize else 1
        adv_sign_grad = adv_examples + direction * alpha * ops.sign(adv_grad)
        adv_examples = ops.stop_gradient(
            clip(adv_examples, adv_sign_grad, clip_eps) if do_clip else adv_sign_grad
        )

    return ops.clip_by_value(adv_examples, 0.0, 1.0)

def train_adv_fgsm(model, loss_fct, adv_examples, adv_targets,
                   epochs=1, alpha=(8 / 255), clip_eps=(8 / 255)):
    return train_adv_examples(
        model, loss_fct, adv_examples, adv_targets,
        epochs=epochs, alpha=alpha, clip_eps=clip_eps, do_clip=True
    )


def train_adv_bim(model, loss_fct, adv_examples, adv_targets, epochs=10,
                  alpha=1.0, clip_eps=(1 / 255) * 8):
    return train_adv_examples(model, loss_fct, adv_examples, adv_targets,
                              epochs=epochs, alpha=alpha, do_clip=True, clip_eps=clip_eps)


def train_adv_least_likely(model, loss_fct, adv_examples, epochs=10,
                           alpha=0.1, clip_eps=(1 / 255) * 8):
    model.set_train(False)
    adv_targets = ops.argmin(model(adv_examples), axis=1)
    return train_adv_examples(model, loss_fct, adv_examples, adv_targets,
                              epochs=epochs, alpha=alpha, do_clip=True,
                              clip_eps=clip_eps, minimize=True)


# 修正后的LeNet模型
class LeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid')  # 关键修正点1：禁用自动padding
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(6 * 14 * 14, 10)  # 输入维度精确匹配

    def construct(self, x):
        x = self.relu(self.conv1(x))  # 输出形状：[B,6,28,28]
        x = self.maxpool(x)  # 输出形状：[B,6,14,14]
        x = self.flatten(x)  # 输出形状：[B,6 * 14 * 14=1176]
        return self.fc(x)  # 矩阵乘法维度：[1176] × [1176,10]


if __name__ == "__main__":
    # 测试模型（维度验证）
    class LeNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
            self.maxpool = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(6 * 14 * 14, 10)

        def construct(self, x):
            x = self.maxpool(nn.ReLU()(self.conv1(x)))
            return self.fc(self.flatten(x))


    # 初始化与测试
    model = LeNet()
    test_input = Tensor(np.random.rand(2, 1, 32, 32), mindspore.float32)
    true_labels = Tensor([3, 7], mindspore.int32)

    # 运行FGSM攻击
    adv_samples = train_adv_examples(model, nn.CrossEntropyLoss(), test_input, true_labels)
    print("扰动范围验证:", ops.abs(adv_samples - test_input).max().asnumpy())  # 应≈0.03137
    print("对抗样本像素范围:", adv_samples.min().asnumpy(), "~", adv_samples.max().asnumpy())  # 应为0.0~1.0
