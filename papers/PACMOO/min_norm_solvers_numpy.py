"""
:author: hqx
"""
import numpy as np

MAX_ITER = 250
STOP_CRIT = 1e-6

class MinNormSolver:
    """[summary]

    Returns:
        [type]: [description]
    """
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0
    def min_norm_element_from2(self, v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        self.counter += 1
        #print('v1v1', v1v1, 'v2v2',v2v2)
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2*v1v2))
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(self, vecs, dps):
        r"""
        Find the minimum norm solution as combination of two points
        This solution is correct if vectors(gradients) lie in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0
        for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)): # for num_tasks:
            for j in range(i+1, len(vecs)):
                #print('vecs[i], vecs[j]', vecs[i], vecs[j])
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    dps[(i, j)] = np.dot(vecs[i], vecs[j])
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    dps[(i, i)] = np.dot(vecs[i], vecs[i])
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    dps[(j, j)] = np.dot(vecs[j], vecs[j])
                c_val, d_val = self.min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d_val < dmin:
                    dmin = d_val
                    sol = [(i, j), c_val, d_val]
        return sol, dps

    def _projection2simplex(self, output):
        r"""
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        self.counter += 1
        length = len(output)
        sorted_y = np.flip(np.sort(output), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(output) - 1.0)/length
        for i in range(length-1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return np.maximum(output - tmax_f, np.zeros(output.shape))

    def _next_point(self, cur_val, grad, num):
        """next_point"""
        proj_grad = grad - (np.sum(grad) / num)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        # skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
        temp = 1
        if tm1[tm1 > 1e-7].shape[0] > 0:
            temp = np.min(tm1[tm1 > 1e-7])
        if tm2[tm2 > 1e-7].shape[0] > 0:
            temp = min(temp, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad*temp + cur_val
        next_point = self._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(self, vecs):
        r"""
        Given a list of vectors (vecs), this method finds the minimum norm element in the
        convex hull as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        vecs[i]是第i个任务的gradients
        It is quite geometric, and the main idea is th e fact that if d_{ij} =
        min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected
        gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = self._min_norm_2d(vecs, dps)
        num = len(vecs) #num_tasks
        sol_vec = np.zeros(num)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if num < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]
        iter_count = 0

        grad_mat = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                grad_mat[i, j] = dps[(i, j)].asnumpy()

        while iter_count < MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = self._next_point(sol_vec, grad_dir, num)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(num):
                for j in range(num):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc_, nd_ = self.min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc_*sol_vec + (1-nc_)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < STOP_CRIT:
                return sol_vec, nd_
            sol_vec = new_sol_vec
        return sol_vec, nd_

    def find_min_norm_element_fw(self, vecs):
        r"""
        Given a list of vectors (vecs), this method finds the minimum norm element
        in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} =
        min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution,
        and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = self._min_norm_2d(vecs, dps)

        num = len(vecs)
        sol_vec = np.zeros(num)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if num < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc_, nd_ = self.min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc_*sol_vec
            new_sol_vec[t_iter] += 1 - nc_

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < STOP_CRIT:
                return sol_vec, nd_
            sol_vec = new_sol_vec
        return sol_vec, nd_


def gradient_normalizers(self, grads, losses, normalization_type):
    """[summary]

    Args:
        grads ([type]): [description]
        losses ([type]): [description]
        normalization_type ([type]): [description]

    Returns:
        [type]: [description]
    """
    self.counter += 1
    gn_ = {}
    num_tsk = len(grads)
    if normalization_type == 'l2':
        for temp in range(num_tsk):
            gn_[temp] = np.sqrt(np.sum([np.sum(np.power(gr, 2)) for gr in grads[temp]]))
    elif normalization_type == 'loss':
        for temp in range(num_tsk):
            gn_[temp] = losses[temp]
    elif normalization_type == 'loss+':
        for temp in range(num_tsk):
            gn_[temp] = losses[temp] * np.sqrt(np.sum([np.sum(np.power(gr, 2)) \
            for gr in grads[temp]]))
    elif normalization_type == 'none':
        for temp in range(num_tsk):
            gn_[temp] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn_
