import mindspore
import mindspore.nn as nn
import mindspore.dataset.vision as vision
from mindspore import Tensor, context
from mindspore import Tensor, ops
from PIL import Image
import os
import mindspore.dataset.transforms as transforms
from s2dnet import S2DNet  

def main():
    # 设置设备
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if mindspore.context.get_context("device_target") == "GPU" else "CPU")
    
    # 假设 checkpoint_path 是模型的预训练权重路径，可选
    checkpoint_path = None

    # 创建 S2DNet 实例
    model = S2DNet(checkpoint_path=checkpoint_path)
    
    # 加载并预处理图像
    image_path = "test_image.jpg"  # 替换为你测试的图像路径
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
    
    # 加载图像并应用相应的转换
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        vision.Resize((224, 224)),  # 根据需要调整大小
        vision.ToTensor(),
    ])
    
    image_tensor = transform(image) 
    image_tensor = Tensor(image_tensor, mindspore.float32)  # 转换后的图像为 [1 x 3 x 224 x 224]
          
    # 模型推理
    model.set_train(False)  # 关闭训练模式
    feature_maps = model.construct(image_tensor)  # 使用construct进行前向传播

    # 打印输出的特征图的尺寸
    print("Shapes of feature maps from the hypercolumn layers:")
    for idx, fmap in enumerate(feature_maps):
        print(f"Feature map {idx}: {fmap.shape}")

if __name__ == "__main__":
    main()
