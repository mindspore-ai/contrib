import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
import ot
import ot.plot


# randomly draw n samples from results
def get_random_subset(results, n):
    results_sample = []
    for array in results:
        if len(array) < n:
            raise ValueError("Sample size n cannot be greater than the number of rows in the input array")
        indices = np.random.choice(array.shape[0], n, replace=False)
        sampled_array = array[indices]
        results_sample.append(sampled_array)
    return results_sample


# compute optimal transport from p1 to p0
def ot_reorganize_xt(xs, xt):
    
    # compute the optimal transport from xs to xt
    # reorganize xt to match xs, i.e. xs[i] is mapped to xt[i] for i=1,...,N
    
    n = len(xs)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
    
    # loss matrix
    M = ot.dist(xs, xt)
    
    # Compute EMD
    G0 = ot.emd(a, b, M)
    
    # reorganize xt
    xt_r = reorganize_xt(xs, xt, G0)
    
    return xt_r

# reorganize samples from p0 to match that of p1
def reorganize_xt(xs, xt, G):
    # Get the number of points (N)
    N = xs.shape[0]
    
    # Initialize the new xt array
    xt_reorganized = np.zeros_like(xt)
    
    # Loop through each row of G to determine the mapping
    for i in range(N):
        # Find the column index j where G[i, j] is non-zero
        j = np.argmax(G[i, :])
        #print(j)
        # Place xt[j] at the position i in xt_reorganized
        xt_reorganized[i, :] = xt[j, :]
    
    return xt_reorganized



# kernel estimation of score
def estimate_density_and_gradient(X, h):
    n_samples, n_features = X.shape
    gradients = np.zeros_like(X)
    
    for j in range(n_samples):
        x = X[j]
        diff = x - X # (x - x1, x-x2, ..., x-xn)
        #print('diff:', diff)
        norm_sq = np.sum(diff**2, axis=1) # (|x-x1|^2, ..., |x-xn|^2)
        #print('norm_sq:', norm_sq)
        weights = np.exp(-norm_sq / (2 * h)) # (e^{-|x-x1|^2/2h}, ..., e^{-|x-xn|^2/2h})
        #print(weights)
        
        q_x = np.sum(weights)
        grad_q_x = np.sum(-diff / h * weights[:, np.newaxis], axis=0)
        #print('test:', weights[:, np.newaxis])
        
        gradients[j] = grad_q_x / q_x
    
    return gradients


# plot vector field
def plot_vectors_from_points(xs, vec_field, plotname='vector field', point_label='Source points nn'):
    """
    Plot the vector field associated with points.

    Parameters:
    xs (N-by-2 array): Source points
    vec_field (N-by-2 array): Vector fields
    """
    
    #mu_t = np.array([2, 2])
    #cov_t = np.array([[1, 0], [0, 1]])


    #xt = ot.datasets.make_2D_samples_gauss(1000, mu_t, cov_t)
    N = xs.shape[0]

    plt.figure(figsize=(10, 8))
    plt.scatter(xs[:, 0], xs[:, 1], s=10, color='orange', label=point_label)
    #plt.scatter(xt[:, 0], xt[:, 1], s=10, color='blue', label='Source true')
    
    plt.quiver(xs[:, 0], xs[:, 1], vec_field[:, 0], vec_field[:, 1], scale=10, scale_units='x', width=0.002, fc='teal')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title(plotname)
    plt.grid(True)
    plt.show()
    
    return



# compute and plot the wasserstein gradient
def wasserstein_gradient(xs, xt, r=1, h=0.2, draw=0):
    
    xt_r = ot_reorganize_xt(xs, xt) # optimal transport map
    
    score = estimate_density_and_gradient(xs, h) # score function
    
    wasserstein_gradient = xs + score - (xt_r - xs) / r # wasserstein gradient 
    
    xi1 = xs + score #\nabla V + \nabla log p = x + score
    xi2 = (xt_r - xs) / r
    
    # calculate the L2(p_{n+1}) norm of the wasserstein gradient
    #error = np.linalg.norm(wasserstein_gradient**2, axis=1)
    #average_error = np.mean(error)
    
    if draw==1:
        plot_vectors_from_points(xs, wasserstein_gradient, plotname='vector field')
    
    #print(f"Average Error: {average_error}")
  
    return wasserstein_gradient, xi1, xi2, score

def l2_norm(x):
    error = np.linalg.norm(x**2, axis=1)
    average_error = np.mean(error)
    
    return average_error


if __name__ == '__main__':

    import pickle
    with open("pushed_data.pkl", "rb") as f:
        data = pickle.load(f)

    results = [v for k, v in data.items()]
    
    n = 2000 # number of samples

    results_sample = get_random_subset(results, n)
    
    x_test = results_sample[0]
    plt.scatter(x_test[:, 0], x_test[:, 1], s=0.01)
    plt.show()