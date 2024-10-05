import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

class MADA(nn.Cell):
    def __init__(self, n_classes, convnet=None, classifier=None):
        super().__init__()

        self._n_classes = n_classes

        self._convnet = convnet or ConvNet()
        self._classifier = classifier or Classifier(n_classes, 12544)
        self._grl = GRL(factor=-1)
        self._domain_classifiers = [
            Classifier(1, 12544)
            for _ in range(n_classes)
        ]

    def construct(self, x):
        features = self._convnet(x)
        features = features.view(features.shape[0], -1)

        logits = self._classifier(features)
        predictions = ops.softmax(logits, axis=1)

        features = self._grl(features)
        domain_logits = []
        for class_idx in range(self._n_classes):
            weighted_features = predictions[:, class_idx].unsqueeze(1) * features
            domain_logits.append(
                self._domain_classifiers[class_idx](weighted_features)
            )

        return logits, domain_logits


class Classifier(nn.Cell):
    def __init__(self, n_classes, input_dimension):
        super().__init__()

        self._n_classes = n_classes
        self._clf = nn.Dense(input_dimension, n_classes)

    def construct(self, x):
        return self._clf(x)


class ConvNet(nn.Cell):
    def __init__(self):
        super().__init__()

        self._convnet = nn.SequentialCell(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def construct(self, x):
        return self._convnet(x)

class GRL(nn.Cell):
    def __init__(self, factor=-1):
        super(GRL, self).__init__()
        self.factor = factor

    def construct(self, x):
        return x  # 前向传播直接返回输入

    def bprop(self, x, out, dout):
        return (self.factor * dout,)
    
def main():
    n_classes = 10
    model = MADA(n_classes=n_classes)
    input_tensor = Tensor(np.random.randn(1, 3, 15, 15), ms.float32)
    logits, domain_logits = model(input_tensor)
    print("Logits shape: ", logits.shape)
    print("Domain logits: ", domain_logits)


if __name__ == "__main__":
    main()