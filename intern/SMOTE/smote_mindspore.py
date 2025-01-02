import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
from random import randint
import random


class SMOTE:
    def __init__(self, distance='euclidian', dims=512, k=5):
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance

    def populate(self, n_samples, i, nnarray, min_samples, k):
        while n_samples:
            nn = randint(0, k - 2)
            diff = min_samples[nnarray[nn]] - min_samples[i]
            gap = random.uniform(0, 1)
            self.synthetic_arr[self.newindex, :] = min_samples[i] + gap * diff
            self.newindex += 1
            n_samples -= 1

    def k_neighbors(self, euclid_distance, k):
        _, idxs = ops.Sort()(euclid_distance)
        return idxs[:, 1:k]

    def find_k(self, x, k):
        euclid_distance = ops.Zeros()((x.shape[0], x.shape[0]), ms.float32)
        for i in range(len(x)):
            dif = (x - x[i]) ** 2
            dist = ops.Sqrt()(dif.sum(axis=1))
            euclid_distance[i] = dist
        return self.k_neighbors(euclid_distance, k)

    def generate(self, min_samples, n_percentage, k):
        t = min_samples.shape[0]
        self.synthetic_arr = ops.Zeros()((int(n_percentage / 100) * t, self.dims), ms.float32)
        n_samples = int(n_percentage / 100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k(min_samples, k)
        for i in range(indices.shape[0]):
            self.populate(n_samples, i, indices[i], min_samples, k)
        self.newindex = 0
        return self.synthetic_arr

    def fit_generate(self, x, y):
        occ = ops.Eye()(int(y.max() + 1), int(y.max() + 1), ms.float32)[y].sum(axis=0)
        dominant_class = ops.Argmax()(occ)
        n_occ = int(occ[dominant_class].asnumpy())
        for i in range(len(occ)):
            if i != dominant_class:
                n_percentage = (n_occ - occ[i]) * 100 / occ[i]
                candidates = x[y == i]
                xs = self.generate(candidates, n_percentage, self.k)
                x = ops.Concat()((x, xs))
                ys = ops.Ones()(xs.shape[0], ms.int32) * i
                y = ops.Concat()((y, ys))
        return x, y


def main():
    x_majority = np.random.randn(100, 2).astype(np.float32)
    y_majority = np.zeros(100, dtype=np.int32)
    x_minority = np.random.randn(20, 2).astype(np.float32)
    y_minority = np.ones(20, dtype=np.int32)

    x = np.concatenate((x_majority, x_minority), axis=0)
    y = np.concatenate((y_majority, y_minority), axis=0)

    x = Tensor(x)
    y = Tensor(y)

    smote = SMOTE(distance='euclidian', dims=2, k=5)

    x_balanced, y_balanced = smote.fit_generate(x, y)

    print("Original dataset shape:", x.shape)
    print("Balanced dataset shape:", x_balanced.shape)
    print("Class distribution after balancing:")
    unique, counts = np.unique(y_balanced.asnumpy(), return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == "__main__":
    main()