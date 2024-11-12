import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from SVGD import SVGD_model

sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )

class OneDimensionGM():

    def __init__(self, omega, mean, var):
        self.omega = omega
        self.mean = mean
        self.var = var

    def dlnprob(self, x):
        rep_x = np.matlib.repmat(x, 1, self.omega.shape[0])
        category_prob = np.exp(- (rep_x - self.mean) ** 2 / (2 * self.var)) / (np.sqrt(2 * np.pi * self.var)) * self.omega
        den = np.sum(category_prob, 1)
        num = ((- (rep_x - self.mean) / self.var) * category_prob).sum(1)
        return np.expand_dims((num / den), 1)

    def MGprob(self, x):
        rep_x = np.matlib.repmat(x, 1, self.omega.shape[0])
        category_prob = np.exp(- (rep_x - self.mean) ** 2 / (2 * self.var)) / (np.sqrt(2 * np.pi * self.var)) * self.omega
        den = np.sum(category_prob, 1)
        return np.expand_dims(den, 1)

if __name__ == "__main__":

    w = np.array([1/3, 2/3])
    mean = np.array([-2, 2])
    var = np.array([1, 1])

    OneDimensionGM_model = OneDimensionGM(w, mean, var)

    np.random.seed(0)

    x0 = np.random.normal(-10, 1, [100, 1]);
    dlnprob = OneDimensionGM_model.dlnprob

    svgd_model = SVGD_model()
    n_iter = 500
    x = svgd_model.update(x0, dlnprob, n_iter=n_iter, stepsize=1e-1, bandwidth=-1, alpha=0.9, debug=True)


    #plot result
    sns.kdeplot(x.reshape((100,)), bw = .4, color = 'g')

    mg_prob = OneDimensionGM_model.MGprob
    x_lin = np.expand_dims(np.linspace(-15, 15, 100), 1)
    x_prob = mg_prob(x_lin)
    plt.plot(x_lin, x_prob, 'b--')
    plt.axis([-15, 15, 0, 0.4])
    plt.title(str(n_iter) + '$ ^{th}$ iteration')
    plt.show()


