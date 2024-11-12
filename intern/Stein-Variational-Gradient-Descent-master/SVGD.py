import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD_model():

    def __init__(self):
        pass

    def SVGD_kernal(self, x, h=-1):
        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = h ** 2 / np.log(x.shape[0] + 1)

        kernal_xj_xi = np.exp(- pairwise_dists ** 2 / h)
        d_kernal_xi = np.zeros(x.shape)
        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = np.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

        return kernal_xj_xi, d_kernal_xi

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        x = np.copy(x0)

        # adagrad with momentum
        eps_factor = 1e-8
        historical_grad_square = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            kernal_xj_xi, d_kernal_xi = self.SVGD_kernal(x, h=-1)
            current_grad = (np.matmul(kernal_xj_xi, lnprob(x)) + d_kernal_xi) / x.shape[0]
            if iter == 0:
                historical_grad_square += current_grad ** 2
            else:
                historical_grad_square = alpha * historical_grad_square + (1 - alpha) * (current_grad ** 2)
            adj_grad = current_grad / np.sqrt(historical_grad_square + eps_factor)
            x += stepsize * adj_grad

        return x

