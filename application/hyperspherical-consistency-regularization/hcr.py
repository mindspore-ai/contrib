import mindspore
from mindspore import ops, nn


class HCR:
    def __init__(self, classifier_network, lr, optimizer_func=nn.Adam, weight=1.0) -> None:
        self.eps = 1e-12
        self.weight = weight
        self.optimizer = optimizer_func(
            classifier_network.trainable_params(), lr)
        self.loss = None

    def pairwise_dist(self, x):
        x_square = x.pow(2).sum(axis=1)
        prod = x @ x.t()
        pdist = (x_square.unsqueeze(1) + x_square.unsqueeze(0) -
                 2 * prod).clamp(min=self.eps)
        pdist[tuple(range(len(x))), tuple(range(len(x)))] = 0.
        return pdist

    def pairwise_prob(self, pdist):
        return ops.exp(-pdist)

    def hcr_loss(self, h, g):
        q1, q2 = self.pairwise_prob(self.pairwise_dist(
            h)), self.pairwise_prob(self.pairwise_dist(g))
        return -1 * (q1 * ops.log(q2 + self.eps)).mean() + -1 * ((1 - q1) * ops.log((1 - q2) + self.eps)).mean()

    def hcr_loss_weight(self, h, g):
        return self.hcr_loss(h, g) * self.weight

    def update(self, logits, projections):
        grad_fn = mindspore.value_and_grad(
            self.hcr_loss_weight, None, self.optimizer.trainable_params())
        self.loss, grads = grad_fn(logits, projections)
        self.optimizer(grads)


class LeNet5(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='pad')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='pad')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.SequentialCell(
            nn.Flatten(1, -1),
            nn.Dense(400, 120)
        )
        self.fc2 = nn.Dense(120, 84)
        self.output = nn.Dense(84, 10)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu(x))
        x = self.conv2(x)
        x = self.pool2(self.relu(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    mindspore.set_context(device_target='CPU')
    mindspore.run_check()

    # define & configure the network
    network = LeNet5()
    optimizer = nn.Adam

    learning_rate = 0.01
    num_epochs = 20
    hcr_reg = HCR(network, learning_rate, optimizer)

    # generate dataset
    BATCH_SIZE = 8
    NUM_CHENNELS = 1
    HEIGHT = 32
    WEIGHT = 32
    NUM_LABELS = 10

    data = ops.rand([BATCH_SIZE, NUM_CHENNELS, HEIGHT, WEIGHT])
    labels = ops.randint(0, 9, (BATCH_SIZE, NUM_LABELS)
                         ).type(mindspore.float32)

    logits = network(data)
    loss = hcr_reg.hcr_loss(logits, labels)
    print(f"loss: {loss}")
