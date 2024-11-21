import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common import dtype as mstype
import pickle
import numpy as np

from gdn import GDN

# 定义带有损失计算的 GDN 前向传播函数
class GDNWithLoss(nn.Cell):
    def __init__(self, gdn_layer):
        super(GDNWithLoss, self).__init__()
        self.gdn_layer = gdn_layer

    def construct(self, input_tensor):
        output = self.gdn_layer(input_tensor)
        loss = ops.mean(output)  # 计算均值损失
        return loss

# 定义测试函数
def test_backward():
    # 从保存的文件加载测试数据
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    input_data = test_data['input'].numpy()  # 转换为 numpy 格式
    expected_gradient = test_data['input_gradient'].numpy()  # 加载预期的梯度

    # 创建带有损失计算的的 GDN 层
    gdn_layer = GDNWithLoss(GDN(ch=3, device='cpu'))
    
    # 转换输入数据为 MindSpore Tensor
    input_tensor = Tensor(input_data, dtype=mstype.float32)

    # 反向传播测试
    grad_fn = ops.GradOperation(get_all=True)
    grad_input = grad_fn(gdn_layer)(input_tensor)[0].asnumpy()

    # 检查反向传播梯度是否一致
    if np.allclose(grad_input, expected_gradient, atol=0.001):
        print("测试通过：MindSpore 梯度与 PyTorch 梯度一致。")
    else:
        print("测试失败：MindSpore 梯度与 PyTorch 梯度不一致。")

    # 输出梯度的相对误差
    grad_relative_error = np.mean(np.abs(grad_input - expected_gradient) / (np.abs(expected_gradient) + 1e-9))
    print(f"梯度相对误差: {grad_relative_error}")

# 定义测试函数
def test_forward():
    # 从保存的文件加载测试数据
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    input_data = test_data['input'].numpy()  # 转换为 numpy 格式
    expected_output = test_data['output'].numpy()

    # 创建 GDN 层
    gdn_layer = GDN(ch=3, device='cpu')
    
    # 转换输入数据为 MindSpore Tensor
    input_tensor = Tensor(input_data, dtype=mstype.float32)

    # 前向传播得到输出
    output_tensor = gdn_layer(input_tensor)

    # 将 MindSpore 的输出转换为 numpy 格式，便于比较
    output_data = output_tensor.asnumpy()

    # 检查前向传播输出是否一致
    if np.allclose(output_data, expected_output, atol=0.001):
        print("测试通过：MindSpore 前向输出与 PyTorch 前向输出一致。")
    else:
        print("测试失败：MindSpore 前向输出与 PyTorch 前向输出不一致。")
    
    # 计算相对误差
    relative_error = np.mean(np.abs(output_data - expected_output) / (np.abs(expected_output) + 1e-9))
    print(f"前向传播相对误差: {relative_error}")


# 主函数
def main():
    # 创建 GDN 层
    gdn_layer = GDN(ch=3, device='cpu')
    
    # 创建一个随机初始化的张量
    input_data = ms.Tensor(shape=(1, 3, 16, 16), dtype=ms.float32, init='normal')

    # 前向传播得到输出
    output_tensor = gdn_layer(input_data)

    print("output_tensor shape:", output_tensor.shape)

if __name__ == '__main__':
    main()
    # test_forward()
    # test_backward()
