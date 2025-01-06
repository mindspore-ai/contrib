import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common.initializer import Normal
import time
import matplotlib.pyplot as plt


def efficient_conv_bn_eval_forward(bn: nn.BatchNorm2d,
                                 conv: nn.Conv2d,
                                 x: Tensor):
    """MindSpore版本的ConvBN优化前向传播。

    Args:
        bn: BatchNorm2d层
        conv: Conv2d层
        x: 输入张量

    Returns:
        优化后的卷积输出
    """
    # 处理卷积和BN的权重和偏置
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = ops.zeros_like(bn.moving_variance)

    if bn.gamma is not None:
        bn_weight = bn.gamma
    else:
        bn_weight = ops.ones_like(bn.moving_variance)

    if bn.beta is not None:
        bn_bias = bn.beta
    else:
        bn_bias = ops.zeros_like(bn.moving_variance)

    # 计算权重系数
    weight_coeff = ops.rsqrt(bn.moving_variance + bn.eps)
    weight_coeff = weight_coeff.reshape((-1,) + (1,) *
                                      (len(conv.weight.shape) - 1))
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # 更新权重和偏置
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    bias_on_the_fly = (bn_bias +
                       coefff_on_the_fly.flatten() *
                       (bias_on_the_fly - bn.moving_mean))

    # 执行卷积操作
    return ops.conv2d(
        x,
        weight_on_the_fly,
        bias_on_the_fly,
        stride=conv.stride,
        pad_mode=conv.pad_mode,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.group
    )


class EfficientConvBN(nn.Cell):
    """ConvBN优化模块。"""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """初始化EfficientConvBN模块。

        Args:
            conv: 卷积层
            bn: 批量归一化层
        """
        super().__init__()
        self.conv = conv
        self.bn = bn

    def construct(self, x):
        """前向传播函数。

        Args:
            x: 输入张量

        Returns:
            处理后的输出张量
        """
        if not self.training:
            return efficient_conv_bn_eval_forward(self.bn, self.conv, x)
        conv_out = self.conv(x)
        return self.bn(conv_out)


class SimpleNet(nn.Cell):
    """测试用的简单网络。"""

    def __init__(self):
        """初始化SimpleNet。"""
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 16, 3,
            padding=1,
            pad_mode='pad',
            has_bias=True
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.efficient_conv_bn = EfficientConvBN(self.conv1, self.bn1)

    def construct(self, x):
        """前向传播函数。

        Args:
            x: 输入张量

        Returns:
            处理后的输出张量
        """
        return self.efficient_conv_bn(x)


def test_different_input_sizes():
    """测试不同输入尺寸的情况。"""
    # 设置运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 创建模型
    model = SimpleNet()
    model.set_train(False)
    
    # 扩展测试范围
    batch_sizes = [1, 2, 4, 8, 16, 32]
    input_sizes = [8, 16, 32, 64, 128, 256]
    
    results = []
    print("\n=== 不同输入尺寸测试结果 ===")
    print("批次大小  输入尺寸  推理时间(ms)  输出均值    输出标准差")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        for input_size in input_sizes:
            # 创建随机输入数据
            input_data = Tensor(
                np.random.randn(batch_size, 3, input_size, input_size),
                ms.float32
            )
            
            # 运行模型并记录时间
            start_time = time.time()
            output = model(input_data)
            inference_time = (time.time() - start_time) * 1000
            
            # 计算统计数据
            output_mean = float(output.mean().asnumpy())
            output_std = float(output.std().asnumpy())
            
            print(f"{batch_size:8d}  {input_size:8d}  {inference_time:11.2f}  "
                  f"{output_mean:10.4f}  {output_std:10.4f}")
            
            results.append({
                "批次大小": batch_size,
                "输入尺寸": input_size,
                "推理时间": inference_time,
                "输出均值": output_mean,
                "输出标准差": output_std
            })
    
    return results

def test_train_eval_modes():
    """测试训练模式和评估模式的区别。"""
    # 设置运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 创建模型
    model = SimpleNet()
    
    # 测试不同的输入尺寸
    input_sizes = [32, 64, 128]
    batch_sizes = [4, 8, 16]
    
    print("\n=== 训练模式与评估模式对比 ===")
    print("批次大小  输入尺寸  训练时间(ms)  评估时间(ms)  模式差异")
    print("-" * 65)
    
    results = []
    for batch_size in batch_sizes:
        for input_size in input_sizes:
            # 创建固定的输入数据用于对比
            np.random.seed(42)
            input_data = Tensor(
                np.random.randn(batch_size, 3, input_size, input_size),
                ms.float32
            )
            
            # 训练模式测试
            model.set_train(True)
            start_time = time.time()
            train_output = model(input_data)
            train_time = (time.time() - start_time) * 1000
            
            # 评估模式测试
            model.set_train(False)
            start_time = time.time()
            eval_output = model(input_data)
            eval_time = (time.time() - start_time) * 1000
            
            # 计算模式差异
            mode_diff = float((train_output - eval_output).abs().mean().asnumpy())
            
            print(f"{batch_size:8d}  {input_size:8d}  {train_time:11.2f}  "
                  f"{eval_time:11.2f}  {mode_diff:10.6f}")
            
            results.append({
                "批次大小": batch_size,
                "输入尺寸": input_size,
                "训练时间": train_time,
                "评估时间": eval_time,
                "模式差异": mode_diff
            })
    
    return results

def print_summary_statistics(eval_results, train_eval_results):
    """打印汇总统计信息。"""
    print("\n=== 性能统计汇总 ===")
    
    # 计算推理时间统计
    inference_times = [r["推理时间"] for r in eval_results]
    print(f"\n推理时间统计 (ms):")
    print(f"最小值: {min(inference_times):.2f}")
    print(f"最大值: {max(inference_times):.2f}")
    print(f"平均值: {np.mean(inference_times):.2f}")
    print(f"中位数: {np.median(inference_times):.2f}")
    
    # 计算训练和评估模式时间差异
    time_diffs = [(r["训练时间"] - r["评估时间"]) for r in train_eval_results]
    print(f"\n训练模式与评估模式时间差异 (ms):")
    print(f"最小差异: {min(time_diffs):.2f}")
    print(f"最大差异: {max(time_diffs):.2f}")
    print(f"平均差异: {np.mean(time_diffs):.2f}")
    
    # 计算模式输出差异统计
    mode_diffs = [r["模式差异"] for r in train_eval_results]
    print(f"\n训练模式与评估模式输出差异:")
    print(f"最小差异: {min(mode_diffs):.6f}")
    print(f"最大差异: {max(mode_diffs):.6f}")
    print(f"平均差异: {np.mean(mode_diffs):.6f}")

def main():
    """主函数。"""
    print("开始性能测试...")
    
    eval_results = test_different_input_sizes()
    train_eval_results = test_train_eval_modes()
    print_summary_statistics(eval_results, train_eval_results)

if __name__ == "__main__":
    import time
    main()
