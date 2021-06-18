import mindspore.nn as N
import mindspore.ops as P


class BLSBasicTrain(N.Cell):
    def __init__(self) -> None:
        super(BLSBasicTrain, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


if __name__ == '__main__':
    bls = BLSBasicTrain()
