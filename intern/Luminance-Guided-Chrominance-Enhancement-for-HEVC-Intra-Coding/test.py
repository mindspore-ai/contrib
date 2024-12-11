import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from network import *

def test_cb_y_net_train_infer():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # 参数设置
    feature_channels = 64
    batch_size = 4
    height, width = 64, 64
    epochs = 1  # 训练轮数

    # 创建网络实例
    model = cb_y_net(feature_channels)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = nn.Adam(model.trainable_params())
    def get_loss(cb,y,target):
        # 前向传播
        output = model(cb, y)
        loss = criterion(output, target)
        return loss
    
    grad_fn = mindspore.value_and_grad(get_loss, None, optimizer.parameters)

    # 生成随机数据
    for epoch in range(epochs):
        cb = ops.randn(batch_size, 1, height // 2, width // 2)  # cb输入
        y = ops.randn(batch_size, 1, height, width)  # y输入
        target = ops.randn(batch_size, 1, height // 2, width // 2)  # 目标输出
        loss, grads = grad_fn(cb,y,target)
        optimizer(grads)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss}")

    print("\nTraining complete.\n")

    # 推断测试
    cb_test = ops.randn(batch_size, 1, height//2, width//2)
    y_test = ops.randn(batch_size, 1, height, width)
    output_test = model(cb_test, y_test)

    # 输出测试结果
    print(f"Test Input cb shape: {cb_test.shape}")
    print(f"Test Input y shape: {y_test.shape}")
    print(f"Test Output shape: {output_test.shape}")

    assert output_test.shape == (batch_size, 1, height // 2, width // 2), (
        f"Expected output shape (B, 1, {height // 2}, {width // 2}), "
        f"but got {output_test.shape}"
    )

    print("Training and inference test passed successfully!")

# 调用测试函数
test_cb_y_net_train_infer()
