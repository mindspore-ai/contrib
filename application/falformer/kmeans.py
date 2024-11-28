import math
from time import time

import mindspore as ms
from mindspore import ops
import numpy as np


def _kpoints(data, k, sample_size=-1):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int (not used)
        sample data to avoid memory overflow during calculation

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """
    return data[ops.randint(0, data.shape[0], size=(k,))]


class KMeans:
    '''
    Kmeans clustering algorithm implemented with MindSpore (refer to https://pypi.org/project/fast-pytorch-kmeans/)

    Parameters:
      n_clusters: int,
        Number of clusters

      max_iter: int, default: 100
        Maximum number of iterations

      tol: float, default: 0.0001
        Tolerance

      verbose: int, default: 0
        Verbosity

      mode: {'euclidean', 'cosine'}, default: 'euclidean'
        Type of distance measure

      init_method: {'random', 'point', '++'}
        Type of initialization

      minibatch: {None, int}, default: None
        Batch size of MinibatchKmeans algorithm
        if None perform full KMeans algorithm

    Attributes:
      centroids: ms.Tensor, shape: [n_clusters, n_features]
        cluster centroids
    '''

    def __init__(self, n_clusters, max_iter=100, tol=0.0001, verbose=0, mode="euclidean", init_method="random",
                 minibatch=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.mode = mode
        self.init_method = init_method
        self.minibatch = minibatch
        self._loop = False
        self._show = False

        try:
            import PYNVML
            self._pynvml_exist = True
        except ModuleNotFoundError:
            self._pynvml_exist = False

        self.centroids = None

    @staticmethod
    def cos_sim(a, b):
        """
          Compute cosine similarity of 2 sets of vectors

          Parameters:
          a: ms.Tensor, shape: [m, n_features]

          b: ms.Tensor, shape: [n, n_features]
        """
        l2_normalize = ops.L2Normalize()
        return l2_normalize(a, dim=-1) @ l2_normalize(b, dim=-1).swapaxes(-2, -1)

    @staticmethod
    def euc_sim(a, b):
        """
          Compute euclidean similarity of 2 sets of vectors

          Parameters:
          a: ms.Tensor, shape: [m, n_features]

          b: ms.Tensor, shape: [n, n_features]
        """
        return 2 * a @ b.swapaxes(-2, -1) - (a ** 2).sum(axis=1)[..., :, None] - (b ** 2).sum(axis=1)[..., None, :]

    def remaining_memory(self):
        import subprocess
        """
        Get remaining memory in GPU for the specified device using nvidia-smi.
    
        Returns:
            remaining (int): The remaining memory in bytes.
        """
        try:
            # Run nvidia-smi to get the memory information for the GPU
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE, encoding='utf-8'
            )
            # Extract the available memory for the given GPU
            memory_free = int(result.stdout.split()[0])  # Memory in MB
            remaining = int(memory_free) * 1024
        except Exception as e:
            raise RuntimeError(f"Failed to get GPU memory: {e}")

        return remaining

    def max_sim(self, a, b):
        """
          Compute maximum similarity (or minimum distance) of each vector
          in a with all of the vectors in b

          Parameters:
          a: ms.Tensor, shape: [m, n_features]

          b: ms.Tensor, shape: [n, n_features]
        """

        batch_size = a.shape[0]
        if self.mode == 'cosine':
            sim_func = self.cos_sim
        elif self.mode == 'euclidean':
            sim_func = self.euc_sim

        if ms.context.get_context("device_target") != 'GPU':
            sim = sim_func(a, b)
            max_sim_v, max_sim_i = sim.max(axis=-1, return_indices=True)
            return max_sim_v, max_sim_i
        else:
            if a.dtype == ms.double or a.dtype == ms.float64:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 8
            if a.dtype == ms.float32:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
            elif a.dtype == ms.half or a.dtype == ms.float32:
                expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
            ratio = math.ceil(expected / self.remaining_memory())
            subbatch_size = math.ceil(batch_size / ratio)
            msv, msi = [], []
            for i in range(ratio):
                if i * subbatch_size >= batch_size:
                    continue
                sub_x = a[i * subbatch_size: (i + 1) * subbatch_size]
                sub_sim = sim_func(sub_x, b)
                sub_max_sim_v, sub_max_sim_i = sub_sim.max(axis=-1, return_indices=True)
                del sub_sim
                msv.append(sub_max_sim_v)
                msi.append(sub_max_sim_i)
            if ratio == 1:
                max_sim_v, max_sim_i = msv[0], msi[0]
            else:
                max_sim_v = ops.cat(msv, axis=0)
                max_sim_i = ops.cat(msi, axis=0)
            return max_sim_v, max_sim_i

    def fit_predict(self, X, centroids=None):
        """
          Combination of fit() and predict() methods.
          This is faster than calling fit() and predict() seperately.

          Parameters:
          X: ms.Tensor, shape: [n_samples, n_features]

          centroids: {ms.Tensor, None}, default: None
            if given, centroids will be initialized with given tensor
            if None, centroids will be randomly chosen from X

          Return:
          labels: ms.Tensor, shape: [n_samples]
        """
        assert isinstance(X, ms.Tensor), "input must be ms.Tensor"
        assert X.dtype in [ms.half, ms.float16, ms.float32, ms.float64, ms.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        batch_size, emb_dim = X.shape
        start_time = time()
        if centroids is None:
            self.centroids = _kpoints(X, self.n_clusters, self.minibatch)
        else:
            self.centroids = centroids
        num_points_in_clusters = ops.ones(self.n_clusters, dtype=X.dtype)
        closest = None
        for i in range(self.max_iter):
            iter_time = time()
            if self.minibatch is not None:
                x = X[np.random.choice(batch_size, size=[self.minibatch], replace=False)]
            else:
                x = X
            closest = self.max_sim(a=x, b=self.centroids)[1]
            matched_clusters, counts = ops.unique_consecutive(closest.sort()[0], return_counts=True)

            c_grad = ops.zeros_like(self.centroids)
            expanded_closest = closest[None].broadcast_to((self.n_clusters, -1))
            mask = (expanded_closest == ops.arange(self.n_clusters)[:, None]).to(X.dtype)
            c_grad = mask @ x / mask.sum(-1)[..., :, None]
            c_grad[c_grad != c_grad] = 0  # remove NaNs

            error = (c_grad - self.centroids).pow(2).sum()
            if self.minibatch is not None:
                lr = 1 / num_points_in_clusters[:, None] * 0.9 + 0.1
            # lr = 1/num_points_in_clusters[:,None]**0.1
            else:
                lr = 1
            num_points_in_clusters[matched_clusters] += counts
            self.centroids = self.centroids * (1 - lr) + c_grad * lr
            if self.verbose >= 2:
                print('iter:', i, 'error:', error.item(), 'time spent:', round(time() - iter_time, 4))
            if error <= self.tol:
                break

        if self.verbose >= 1:
            print(
                f'used {i + 1} iterations ({round(time() - start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')
        return closest

    def predict(self, X):
        """
          Predict the closest cluster each sample in X belongs to

          Parameters:
          X: ms.Tensor, shape: [n_samples, n_features]

          Return:
          labels: ms.Tensor, shape: [n_samples]
        """
        assert isinstance(X, ms.Tensor), "input must be ms.Tensor"
        assert X.dtype in [ms.half, ms.float16, ms.float32, ms.float64, ms.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        return self.max_sim(a=X, b=self.centroids)[1]

    def fit(self, X, centroids=None):
        """
          Perform kmeans clustering

          Parameters:
          X: ms.Tensor, shape: [n_samples, n_features]
        """
        assert isinstance(X, ms.Tensor), "input must be ms.Tensor"
        assert X.dtype in [ms.half, ms.float16, ms.float32, ms.float64, ms.double], "input must be floating point"
        assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "

        self.fit_predict(X, centroids)
