import mindspore.ops as ops
from mindspore import Tensor, float32
import mindspore.context as context
import mindspore
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


def solver_delta_enhanced_sparse(Y, F, Ms, beta, gamma, delta, MAXITER, step_size):
    # Y is n x f feature matrix. Nonnegative.
    # F is n x k block allocation matrix. Binary.
    # Ms is k x k image/mixing matrix of the raw network topology.
    # beta is the ratio of KL-divergence term.
    # delta is minor postive scalar to avoid trivial solution.
    # MAXITER is a large integer for the maximum amount of iterations.
    # step_size is the step size for gradient descent
    F = Tensor(F, dtype=float32)
    Y = Tensor(Y, dtype=float32)
    Ms = Tensor(Ms, float32)
    beta = Tensor(beta)
    gamma = Tensor(gamma)
    delta = Tensor(delta)

    [n, f] = Y.shape
    k = F.shape[1]
    D = ops.diag(ops.norm(F, dim=0) ** 2)
    Ybar = ops.matmul(ops.matmul(F, D.inverse()), ops.matmul(F.T, Y))
    Dbar = ops.matmul(D.inverse(), ops.matmul(F.T, Y))
    Ms = Ms + ops.ones(k) * delta
    P = ops.zeros((k, k), float32)
    for i in range(k):
        P[i, :] = Ms[i, :] / ops.sum(Ms[i, :])

    Ls_dr = (ops.ones(f) / ops.norm(
        ops.ones(f))).float()  # For sparsity regularization. Weighted by float scalar gamma.
    r = ops.ones(f)  # Make sure it is a vector, not a matrix. Otherwise ops.diag(r) wouldn't work for R.
    for iteration in range(MAXITER):
        print('inside iteraction:' + str(iteration + 1))
        R = ops.diag(r)
        M = ops.matmul(ops.matmul(Dbar, R), Dbar.T) + ops.ones((k, k)) * delta
        A = ops.matmul(ops.matmul(Y, R), Y.T)
        X = ops.matmul(ops.matmul(Ybar, R), Ybar.T)
        YAY = ops.matmul(ops.matmul(Y.T, A), Y)
        Lb = (ops.norm((A - X)) ** 2) / (ops.norm(A) ** 2)
        Lb_dr = mindspore.numpy.diag(YAY + ops.matmul(ops.matmul(Ybar.T, X - (A * 2)), Ybar)) * 2 / (
                ops.norm(A) ** 2) - mindspore.numpy.diag(YAY) * 2 * Lb / (ops.norm(A) ** 2)
        Lm_dr = ops.zeros(f, float32)
        row_sums = ops.sum(M, dim=1, keepdim=True)
        Q = M / row_sums
        for i in range(f):
            Q_rl = ops.matmul(ops.diag(ops.div(ops.ones(k), ops.sum(M, 1))), (
                    (ops.outer(Dbar[:, i], Dbar[:, i])) - ops.matmul(ops.diag(Dbar[:, i]), Q) * ops.sum(
                Dbar[:, i])))
            Lm_dr[i] = ops.trace(ops.matmul((ops.log(ops.div(Q, P)) + P).T, Q_rl))
        L_dr = (Lb_dr / ops.norm(Lb_dr)) * (Tensor(1.0) - beta) + (Lm_dr / ops.norm(Lm_dr)) * beta + Ls_dr * gamma
        r = ops.maximum(r - L_dr * step_size, Tensor(0.0))
        normr = ops.norm(r)
        r = r / normr
    return r


if __name__ == "__main__":
    datastr = './dataset/'
    resultstr = './results/'
    print('Loading raw graph...')
    E = np.load(datastr + 'BlogCatalog_Network.npz')['E']
    print('Loading nodal attributes...')
    Y = np.load(datastr + 'BlogCatalog_Attributes.npz')['Y']
    kmin = 6
    kmax = 6
    repmax = 10
    gammas = [1, 2, 3]  # Grid-search
    betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    MAXITER = 200  # 200 iterations.
    step_size = 1e-2
    delta = 1e-6
    repselected = 3  # Lowest RRE.
    for k in range(kmin, kmax + 1):  # Iterate from k = kmin to k = kmax
        kstr = str(k)
        # At each k, we first find which block model achieves the lowest reconstruction loss
        rep = repselected
        repstr = str(rep)  # Already aligned rep to start at 1. No need to add 1 to translate to matlab indexing.
        print('Loading block allocation ...')
        F = np.load(datastr + 'blogcatalog_nmtf_' + kstr + '_' + repstr + '_F.npz')['F']
        print('Loading mixing/image matrix ...')
        Ms = np.load(datastr + 'blogcatalog_nmtf_' + kstr + '_' + repstr + '_M.npz')['Ms']
        for beta in betas:
            for gamma in gammas:
                r = solver_delta_enhanced_sparse(Y, F, Ms, beta, gamma, delta, MAXITER, step_size)
                betastr = '0' + str(beta * 10)
                np.savetxt(resultstr + 'ours_blogcatalog_k_' + kstr + '_beta_' + betastr + '_gamma_' + str(
                    gamma) + '_epoch_' + str(MAXITER) + '_r.csv', (r.asnumpy()), delimiter=',')
