import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from nwt_mindspore import Memcodes
import time
import os


def test_memcodes(small_test=True):
    """测试Memcodes模块的基本功能"""
    print("开始测试Memcodes模块...")

    # 设置递归深度限制
    print("设置递归深度限制为 3000")
    ms.set_recursion_limit(3000)

    # 设置MindSpore上下文
    print("设置MindSpore上下文为PYNATIVE_MODE以便调试")
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    # 创建Memcodes实例
    if small_test:
        # 使用较小的尺寸进行初始测试
        dim = 64
        heads = 4
        num_codes = 32
        seq_len = 16
    else:
        # 使用原始论文参数
        dim = 512
        heads = 8
        num_codes = 1024
        seq_len = 32  # 使用较小序列长度以加快测试

    print(f"创建Memcodes: dim={dim}, heads={heads}, num_codes={num_codes}")
    codebook = Memcodes(
        dim=dim,
        heads=heads,
        num_codes=num_codes,
        temperature=1.0
    )

    # 创建随机输入
    batch_size = 1
    print(f"创建输入张量: shape=({batch_size}, {seq_len}, {dim})")
    x = Tensor(np.random.randn(batch_size, seq_len, dim).astype(np.float32))

    # 设置为评估模式
    codebook.set_train(False)

    # 前向传播
    print("执行前向传播...")
    start_time = time.time()
    try:
        out, codebook_indices = codebook(x)
        forward_time = time.time() - start_time
        print(f"前向传播成功: 耗时={forward_time:.4f}秒")
        print(f"输出形状={out.shape}, 编码索引形状={codebook_indices.shape}")

        # 从索引重建输出
        print("尝试从索引重建输出...")
        start_time = time.time()
        reconstructed = codebook.get_codes_from_indices(codebook_indices)
        reconstruct_time = time.time() - start_time
        print(f"重建成功: 重建输出形状={reconstructed.shape}")

        # 检查重建是否与原始输出相同
        print(f"原始输出形状: {out.shape}")
        print(f"重建输出形状: {reconstructed.shape}")

        if out.shape != reconstructed.shape:
            print(f"警告: 形状不匹配 - 原始: {out.shape}, 重建: {reconstructed.shape}")
            return {
                "success": False,
                "error": f"Shape mismatch - original: {out.shape}, reconstructed: {reconstructed.shape}"
            }

        diff = np.abs(out.asnumpy() - reconstructed.asnumpy()).mean()
        print(f"原始输出与重建输出的平均绝对误差: {diff}")

        if diff < 1e-5:
            print("测试通过: 重建输出与原始输出匹配")
        else:
            print(f"测试失败: 重建输出与原始输出不匹配，差异={diff}")

        # 返回结果
        return {
            "success": diff < 1e-5,
            "output_shape": out.shape,
            "indices_shape": codebook_indices.shape,
            "reconstruction_error": diff,
            "forward_time": forward_time,
            "reconstruct_time": reconstruct_time
        }

    except Exception as e:
        print(f"测试失败，错误信息: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def test_full_scale():
    """使用完整规模的模型进行测试，接近原始论文参数"""
    print("\n测试完整规模模型...")

    dim = 512
    heads = 8
    num_codes = 1024

    print(f"创建完整规模Memcodes: dim={dim}, heads={heads}, num_codes={num_codes}")
    codebook = Memcodes(
        dim=dim,
        heads=heads,
        num_codes=num_codes,
        temperature=1.0
    )

    # 使用与论文相同的尺寸进行测试
    batch_size = 1
    seq_len = 1024
    print(f"创建输入张量: shape=({batch_size}, {seq_len}, {dim})")
    x = Tensor(np.random.randn(batch_size, seq_len, dim).astype(np.float32))

    # 设置为评估模式
    codebook.set_train(False)

    # 前向传播
    print("执行前向传播...")
    start_time = time.time()
    try:
        out, codebook_indices = codebook(x)
        forward_time = time.time() - start_time
        print(f"前向传播成功: 耗时={forward_time:.4f}秒")
        print(f"输出形状={out.shape}, 编码索引形状={codebook_indices.shape}")

        # 从索引重建输出
        print("尝试从索引重建输出...")
        start_time = time.time()
        reconstructed = codebook.get_codes_from_indices(codebook_indices)
        reconstruct_time = time.time() - start_time
        print(f"重建成功: 重建输出形状={reconstructed.shape}")

        # 检查重建是否与原始输出相同
        diff = np.abs(out.asnumpy() - reconstructed.asnumpy()).mean()
        print(f"原始输出与重建输出的平均绝对误差: {diff}")

        return {
            "success": diff < 1e-5,
            "output_shape": out.shape,
            "indices_shape": codebook_indices.shape,
            "reconstruction_error": diff,
            "forward_time": forward_time,
            "reconstruct_time": reconstruct_time
        }

    except Exception as e:
        print(f"测试失败，错误信息: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """主函数，运行所有测试"""
    print("=" * 50)
    print("NWT MindSpore 测试")
    print("=" * 50)

    # 测试小型模型
    print("\n=== 小型模型测试 ===")
    small_test_result = test_memcodes(small_test=True)

    # 测试中型模型
    if small_test_result.get("success", False):
        print("\n=== 中型模型测试 ===")
        medium_test_result = test_memcodes(small_test=False)
    else:
        medium_test_result = {"success": False, "error": "跳过中型模型测试，因为小型测试失败"}

    # 测试完整规模模型
    if medium_test_result.get("success", False):
        print("\n=== 完整规模模型测试 ===")
        full_test_result = test_full_scale()
    else:
        full_test_result = {"success": False, "error": "跳过完整规模测试，因为中型测试失败"}

    print("\n总结:")

    if small_test_result.get("success", False):
        print("✓ 小型模型测试通过!")
        print(f"- 输出形状: {small_test_result['output_shape']}")
        print(f"- 索引形状: {small_test_result['indices_shape']}")
        print(f"- 重建误差: {small_test_result['reconstruction_error']:.8f}")
    else:
        print("✗ 小型模型测试失败!")
        print(f"- 错误: {small_test_result.get('error', '未知错误')}")

    if medium_test_result.get("success", False):
        print("\n✓ 中型模型测试通过!")
        print(f"- 输出形状: {medium_test_result['output_shape']}")
        print(f"- 索引形状: {medium_test_result['indices_shape']}")
        print(f"- 重建误差: {medium_test_result['reconstruction_error']:.8f}")
    else:
        print("\n✗ 中型模型测试失败!")
        print(f"- 错误: {medium_test_result.get('error', '未知错误')}")

    if full_test_result.get("success", False):
        print("\n✓ 完整规模模型测试通过!")
        print(f"- 输出形状: {full_test_result['output_shape']}")
        print(f"- 索引形状: {full_test_result['indices_shape']}")
        print(f"- 重建误差: {full_test_result['reconstruction_error']:.8f}")
        print(f"- 前向传播时间: {full_test_result['forward_time']:.4f}秒")
        print(f"- 重建时间: {full_test_result['reconstruct_time']:.4f}秒")
    else:
        print("\n✗ 完整规模模型测试失败!")
        print(f"- 错误: {full_test_result.get('error', '未知错误')}")

    print("=" * 50)

    # 返回所有测试成功
    return small_test_result.get("success", False) and medium_test_result.get("success",
                                                                              False) and full_test_result.get("success",
                                                                                                              False)


if __name__ == "__main__":
    main()