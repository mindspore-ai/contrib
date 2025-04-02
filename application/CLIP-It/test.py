import numpy as np
import mindspore as ms
from mindspore import context, Tensor
import os

from clip_it import CLIP_IT_MS


def test_clip_it_ms():
    # 设置MindSpore运行环境
    # 使用set_device替代context.set_context中的device_target
    ms.set_context(mode=context.PYNATIVE_MODE)
    ms.set_device("CPU")

    # 模型参数
    clip_model_name = "mock_clip_model"  # 仅作为标识符
    num_sentences = 7
    lgattn_n_heads = 2  # 减少注意力头数
    transformer_kwargs = {
        'batch_first': True,
    }

    print("正在初始化CLIP-It MindSpore模型...")
    # 创建CLIP-It MindSpore模型，使用更小的特征维度以加快测试
    model = CLIP_IT_MS(
        clip_model_name,
        num_sentences,
        lgattn_n_heads,
        transformer_kwargs,
        dim=64  # 使用更小的维度加快测试并减少内存使用
    )

    print("创建测试数据...")
    # 创建假的视频数据 (B, C, F, H, W)，使用更小的尺寸
    batch_size = 1
    num_frames = 2  # 减少帧数
    frame_height = 64  # 减少高度
    frame_width = 64  # 减少宽度
    channels = 3

    # 创建随机视频帧数据
    videos = np.random.rand(batch_size, channels, num_frames, frame_height, frame_width).astype(np.float32)

    # 转换为MindSpore Tensor
    videos_ms = Tensor(videos, ms.float32)

    # 创建假的文本数据 - 这里实际上只是占位符
    texts = [
        "a person walking in the park",
        "a car driving on the road",
        "a dog playing with a ball",
        "people sitting in a cafe",
        "a sunset over the mountains",
        "children playing in the playground",
        "birds flying in the sky"
    ]

    # 前向传播
    try:
        print("开始测试CLIP-It MindSpore模型...")
        out = model(videos_ms, texts)
        print(f"输出形状: {out.shape}")
        print(f"输出值: {out}")
        print("测试成功！")
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_clip_it_ms()