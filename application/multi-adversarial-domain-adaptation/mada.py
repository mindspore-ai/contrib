import mindspore
from mindspore import nn, ops


class MADA(nn.Cell):
    def __init__(self, n_classes, convnet=None, classifier=None):
        super().__init__()

        self._n_classes = n_classes
        self._convnet = convnet or ConvNet()
        self._classifier = classifier or Classifier(n_classes, 61504)
        self._grl = GRL(factor=-1)
        self._domain_classifiers = [
            Classifier(1, 61504)
            for _ in range(n_classes)
        ]

    def construct(self, x):
        features = self._convnet(x)
        features = features.reshape(features.shape[0], -1)

        logits = self._classifier(features)
        predictions = ops.softmax(logits, axis=1)

        features = self._grl(features)
        domain_logits = []
        for class_idx in range(self._n_classes):
            weighted_features = predictions[:, class_idx].expand_dims(1) * features
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
        super().__init__()
        self._factor = factor

    def construct(self, x):
        return x

    def backward(self, grad):
        return self._factor * grad

if __name__ == '__main__':
    n_classes = 10
    model = MADA(n_classes)
    batch_size = 4
    input_tensor = ops.randn(batch_size, 3, 32, 32)
    logits, domain_logits = model(input_tensor)
    print("Logits:", logits)
    print("Domain Logits:", domain_logits)