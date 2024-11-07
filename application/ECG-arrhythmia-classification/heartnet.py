import mindspore.nn as nn
import numpy as np
import mindspore


class HeartNet(nn.Cell):
    def __init__(self, num_classes=7):
        super(HeartNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.ELU(),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.ELU(),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.ELU(),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.ELU(),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.ELU(),
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True),
            nn.ELU(),
            nn.BatchNorm2d(256, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.SequentialCell(
            nn.Dropout(p=0.5),
            nn.Dense(16 * 16 * 256, 2048, weight_init="xavier_uniform"),
            nn.ELU(),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Dense(2048, num_classes, weight_init="xavier_uniform"),
        )
        self.set_train(True)

    def construct(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], 16 * 16 * 256)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 32
    channels = 1
    height = 128
    width = 128
    model = HeartNet()
    input_array = np.random.rand(batch_size, channels, height, width)
    input_tensor = mindspore.Tensor(input_array, mindspore.float32)
    output = model(input_tensor)
    print(output.shape)
    print(output)
