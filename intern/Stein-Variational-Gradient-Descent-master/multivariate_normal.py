import numpy as np
from SVGD import SVGD_model

class MVN():

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def dlnprob(self, x):
        return -1 * np.matmul((x - self.mean), np.linalg.inv(self.cov))

if __name__ == "__main__":

    # cov = np.array([[5.3838, -1.3120],[-1.3120, 1.7949]])
    cov = np.array([[0.2260, 0.1652], [0.1652, 0.6779]])
    mean = np.array([-0.6871,0.8010])

    mvn_model = MVN(mean, cov)

    np.random.seed(0)

    x0 = np.random.normal(0, 1, [10, 2]);
    dlnprob = mvn_model.dlnprob

    svgd_model = SVGD_model()
    x = svgd_model.update(x0, dlnprob, n_iter=1000, stepsize=1e-2, bandwidth=-1, alpha=0.9, debug=True)

    print ("Mean ground truth: ", mean)
    print ("Mean obtained by svgd: ", np.mean(x, axis=0))

    print("Cov ground truth: ", cov)
    print("Cov obtianed by svgd: ", np.cov(x.T))
