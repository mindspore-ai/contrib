import numpy as np
import os
import argparse
import cv2
import sys
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import mindspore as ms
from mindspore import Tensor
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from fdcam_mindspore import FDCAM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='../images/henFence.jpg',
                        help='输入图像路径')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='特征余弦相似度阈值')
    parser.add_argument('--output-dir', type=str, default='./results_ms',
                        help='输出结果保存目录')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    return args

def load_image(path):
    """加载并返回原始图像，用于显示"""
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.asarray(img)
    return img

def preprocess_image(img_path):
    """预处理图像为模型输入格式"""
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transform_normalize
    ])
    
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    
    return tensor

class ModelAdapter:
    """简化的PyTorch模型适配器，专为MindSpore FDCAM设计"""
    def __init__(self, pytorch_model, target_layers):
        self.pytorch_model = pytorch_model
        self.target_layers = target_layers
        self.device = next(pytorch_model.parameters()).device
        
        # 在初始化时注册钩子
        self.activations = {}
        self.hooks = []
        
        # 注册前向钩子
        def forward_hook(module, input, output):
            self.activations['value'] = output.detach().cpu().numpy()
        
        for layer in target_layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
    
    def __call__(self, x):
        """处理前向传播"""
        # 检查输入类型
        if isinstance(x, ms.Tensor):
            # 转换为PyTorch tensor
            x = torch.tensor(x.asnumpy()).to(self.device)
        
        # 前向传播
        output = self.pytorch_model(x)
        
        # 返回结果
        if isinstance(output, torch.Tensor):
            return ms.Tensor(output.detach().cpu().numpy())
        return output
    
    def register_forward_hook(self, hook_fn):
        """MindSpore API兼容的钩子注册方法"""
        class HookAdapter:
            def __init__(self, adapter, hook_fn):
                self.adapter = adapter
                self.hook_fn = hook_fn
                
                # 保存原始钩子和激活值
                self.original_activations = adapter.activations
                
                # 注册新的前向钩子
                def forward_hook(module, input, output):
                    # 保存激活值
                    adapter.activations['value'] = output.detach().cpu().numpy()
                    # 模拟MindSpore钩子调用
                    hook_fn(adapter, None, ms.Tensor(output.detach().cpu().numpy()))
                
                # 清除旧钩子
                for h in adapter.hooks:
                    h.remove()
                
                # 注册新钩子
                adapter.hooks = []
                for layer in adapter.target_layers:
                    adapter.hooks.append(layer.register_forward_hook(forward_hook))
                    
            def remove(self):
                """移除钩子"""
                # 恢复原始钩子
                for h in self.adapter.hooks:
                    h.remove()
                self.adapter.hooks = []
                self.adapter.activations = self.original_activations
        
        # 返回适配器
        return HookAdapter(self, hook_fn)

def generate_cam(fdcam, model, target_layers, input_tensor, targets):
        # 1. 获取激活和梯度
        activations = []
        gradients = []
        
        # 前向传播，获取激活值
        _ = model(input_tensor)
        activations = model.activations.get('value', None)
        if activations is None:
            # 无法获取激活值，返回全黑图像
            return np.zeros((1, input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)
            
        # 简化的"梯度"计算 - 使用激活统计信息
        gradients = np.mean(activations, axis=(2, 3), keepdims=True)
        gradients = np.repeat(gradients, activations.shape[2], axis=2)
        gradients = np.repeat(gradients, activations.shape[3], axis=3)
        
        # 使用fdcam计算权重
        weights = fdcam.get_cam_weights(input_tensor, target_layers[0], targets, activations, gradients)
        
        # 生成类激活图
        cam = np.zeros((activations.shape[2], activations.shape[3]), dtype=np.float32)
        for i in range(weights.shape[1]):
            cam += weights[0, i] * activations[0, i, :, :]
            
        # 应用ReLU并归一化
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # 调整大小以匹配输入
        if (cam.shape[0], cam.shape[1]) != (input_tensor.shape[2], input_tensor.shape[3]):
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        # 添加批次维度
        return np.expand_dims(cam, axis=0)

def main():
    args = get_args()
    
    # 设置MindSpore上下文
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 加载PyTorch模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    except:
        model = torchvision.models.vgg16(pretrained=True)
    
    model.eval()
    model = model.to(device)
    
    # 获取目标层 - VGG16的最后一个卷积层
    target_layers = [model.features[-1]]
    
    # 加载和处理图像
    input_tensor = preprocess_image(args.image_path)
    input_tensor = input_tensor.to(device)
    img_show = load_image(args.image_path)
    
    # 获取预测的类别
    output = model(input_tensor)
    target_category = output.argmax(dim=1).item()
    print(f"预测类别ID: {target_category}")
    
    # 创建适配器
    model_adapter = ModelAdapter(model, target_layers)
    
    # 创建FDCAM实例
    fdcam = FDCAM(
        model=model_adapter,
        target_layers=[model_adapter],  # 直接使用适配器作为目标层
        threshold=args.threshold
    )
    
    # 创建类别目标
    targets = [ClassifierOutputTarget(target_category)]
    
    # 转换输入张量为MindSpore格式
    ms_input_tensor = ms.Tensor(input_tensor.cpu().numpy())
    
    # 生成热力图
    grayscale_cam = generate_cam(fdcam, model_adapter, target_layers, ms_input_tensor, targets)       
    grayscale_cam = grayscale_cam[0, :]
    
    # 可视化并保存结果
    visualization = show_cam_on_image(img_show/255, grayscale_cam, use_rgb=True)
    
    # 显示结果
    plt.figure(figsize=(12, 5))
    
    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_show)
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示CAM可视化结果
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('FD-CAM Visualization')
    plt.axis('off')
    
    # 保存结果
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    plt.savefig(output_path)
    
    # 保存热力图
    cv2.imwrite(output_path.replace('.jpg', '_heatmap.jpg'), np.uint8(255 * grayscale_cam))
    
    print(f"结果已保存到: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()