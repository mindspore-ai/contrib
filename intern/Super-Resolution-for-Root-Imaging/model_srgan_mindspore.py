import mindspore.nn as nn

class AutoGen2(nn.Cell):
    # 保持原有的生成器类实现
    def __init__(self, ngpu=1, nc_in=1, nc_out=1, scale_factor=4):
        super(AutoGen2, self).__init__()
        self.ngpu = ngpu
        self.layers = nn.SequentialCell([
            nn.Conv2d(nc_in, 64, 9, stride=1, pad_mode='pad', padding=4, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, stride=1, pad_mode='pad', padding=2, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(32, nc_out, 5, stride=1, pad_mode='pad', padding=2, has_bias=False)
        ])

    def construct(self, x):
        return self.layers(x)

class Discriminator2(nn.Cell):
    # 保持原有的判别器类实现
    def __init__(self, nc=1, crop_size=64):
        super(Discriminator2, self).__init__()
        self.net = nn.SequentialCell([
            nn.Conv2d(nc, 64, 9, stride=1, pad_mode='pad', padding=4, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, stride=1, pad_mode='pad', padding=2, has_bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, nc, 5, stride=1, pad_mode='pad', padding=2, has_bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        ])
        self.fc = nn.Conv2d(1, 1, crop_size//4, has_bias=False)

    def construct(self, x):
        batch_size = x.shape[0]
        x = self.net(x)
        x = self.fc(x)
        return x.reshape((batch_size, -1))

def main():
    # 初始化模型实例
    generator = AutoGen2(ngpu=1, nc_in=1, nc_out=1)
    discriminator = Discriminator2(nc=1, crop_size=64)
    
    # 打印模型结构
    print("="*50)
    print("Generator Architecture:")
    print(generator)
    print("\nDiscriminator Architecture:")
    print(discriminator)
    print("="*50)
    
    # 计算并打印参数数量
    def count_params(model):
        return sum(p.size for p in model.get_parameters())
    
    print(f"Generator Parameters: {count_params(generator):,}")
    print(f"Discriminator Parameters: {count_params(discriminator):,}")
    print("="*50)

if __name__ == "__main__":
    main()
