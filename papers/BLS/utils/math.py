import mindspore.numpy as mnp
import mindspore.ops as ops


def pinv(mat, reg):
    aat = mnp.matmul(mnp.transpose(mat), mat)
    eye = mnp.eye(mat.shape[1], mat.shape[1])
    res = reg * eye
    res = res + aat
    res = pinv_svd(res)
    res = mnp.matmul(res, mnp.transpose(mat))
    return res


def orthonormalize(_matrix):
    q, _ = qr_decomposition(_matrix)
    return q


def qr_decomposition(mat):
    (m, n) = mat.shape
    r = mat
    q = mnp.eye(m, m)
    for j in range(n):
        # Apply Householder transformation.
        v = r[:, j]
        if j > 0:
            v[0:j] = 0.0
        v = mnp.expand_dims(v, axis=1)
        x1 = mnp.norm(v[j:])
        v = v / (v[j] + mnp.copysign(x1, v[j]))
        v[j] = 1.0
        tau = 2.0 / mnp.matmul(mnp.transpose(v), v)
        h = mnp.eye(m, m)
        h_part = tau * mnp.matmul(v, mnp.transpose(v))
        h_complete = h_part
        h -= h_complete
        r = mnp.matmul(h, r)
        q = mnp.matmul(h, q)
    return mnp.transpose(q[:n]), r[:n]


def pinv_svd(mat):
    u, s, vt = svd(mat)
    s = mnp.reciprocal(s)
    res = mnp.matmul(mnp.transpose(vt), s * mnp.transpose(u))
    return res


def diag(mat):
    (n, m) = mat.shape
    s = []
    for i in range(m):
        s.append(mat[i][i])
    sigma = mnp.stack(s, axis=0)
    return sigma


def svd(mat):
    sigma = 0.0
    u = []
    (a, b) = mat.shape
    v = mnp.eye(b, b)
    for i in range(3):
        u, _ = qr_decomposition(mnp.matmul(mat, v))
        v, s = qr_decomposition(mnp.matmul(mnp.transpose(mat), u))
        sigma = mnp.diag(s)
    return u, mnp.expand_dims(sigma, axis=1), mnp.transpose(v)


def map_minmax(matrix, v_min=-1.0, v_max=1.0):
    max_list = mnp.amax(matrix, axis=0)
    min_list = mnp.amin(matrix, axis=0)
    max_min_dist = max_list - min_list
    mat_min_dist = matrix - min_list
    std = mat_min_dist / max_min_dist
    temp_scare = std * (v_max - v_min)
    x_scale = temp_scare + v_min
    return x_scale, max_list, min_list


def sparse_bls(self, z, x):
    iter = 50
    lam = 0.001
    m = z.shape[1]
    n = x.shape[1]
    wk = ok = uk = mnp.full((m, n), 0.0)
    # L1 = self.matmul_ta_op(Z, Z) + self.eye_op(m, m, dtype.float32)
    l1 = mnp.matmul(mnp.transpose(z), z) + mnp.eye(m, m)
    l1_inv = self.pinv_svd(l1)
    l2 = mnp.matmul(mnp.matmul(l1_inv, mnp.transpose(z)), x)
    for i in range(iter):
        ck = l2 + mnp.matmul(l1_inv, ok - uk)
        ok = self.shrinkage(ck + uk, lam)
        cp = ck - ok
        uk = uk + cp
        wk = ok
    return mnp.transpose(wk)


def shrinkage(a, b):
    output = mnp.maximum(a - b, mnp.array([0.0])) - mnp.maximum(-a - b, mnp.array([0.0]))
    return output


def standardize_input(data):
    data = data / 255.0
    normalize = ops.LayerNorm()
    z_score, _, _ = normalize(data, mnp.full((data.shape[1],), 1),
                              mnp.full((data.shape[1],), 0))
    return z_score
