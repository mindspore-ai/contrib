## 运行
1. 直接执行 `python main.py` 即可运行代码。

## 验证与Pytorch-GDN的功能一致性
1. 基于仓库https://github.com/jorge-pessoa/pytorch-gdn，用以下脚本生成测试样例

```python
import torch
import pickle

# 定义生成测试数据的函数
def generate_test_data():
    # 输入张量大小可以根据需求调整
    input_data = torch.randn(1, 3, 16, 16, requires_grad=True)  # 假设输入是一个4D张量

    # 初始化 GDN 层（根据实际情况调整参数）
    gdn_layer = GDN(ch=3, device='cpu')

    # 前向传播得到输出
    output_data = gdn_layer(input_data)

    # 创建一个标量损失函数，例如输出的均值，方便反向传播
    loss = output_data.mean()
    # 进行反向传播以计算输入的梯度
    loss.backward()

    # 获取输入的梯度
    input_gradient = input_data.grad

    # 保存输入数据、前向输出和反向传播的梯度
    with open('test_data.pkl', 'wb') as f:
        pickle.dump({
            'input': input_data.detach(),  # 输入张量
            'output': output_data.detach(),  # 前向输出
            'input_gradient': input_gradient.detach()  # 输入梯度
        }, f)
    print(input_gradient)
    print("测试数据已保存到 'test_data.pkl' 文件中。")

# 执行生成测试数据的函数
generate_test_data()
```

2. 测试样例放入本仓库目录后，在main.py中调用test_forward()函数和test_backward()函数进行验证。