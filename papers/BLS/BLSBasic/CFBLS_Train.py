import argparse
import os
# import moxing as mox
from mindspore import Tensor, dtype
from mindspore.train.serialization import export, save_checkpoint
import mindspore.context as context
import mindspore.ops.operations as P
import mindspore.nn as N
import numpy as np


class CFBLSBasicTrain(N.Cell):
    def __init__(self) -> None:
        super(CFBLSBasicTrain, self).__init__()
        self.s = 0.8
        self.c = 2 ** -15
        self.N1 = 10
        self.N2 = 10
        self.N3 = 20
        self.ymax = 1
        self.ymin = 0
        self.iterations = 2
        self.matmul_op = P.MatMul()
        self.matmul_ta_op = P.MatMul(transpose_a=True)
        self.matmul_tb_op = P.MatMul(transpose_b=True)
        self.stack_v_op = P.Pack(axis=1)
        self.stack_h_op = P.Pack(axis=0)
        self.diag_op = P.Diag()
        self.diag_part_op = P.DiagPart()
        self.eye_op = P.Eye()
        self.transpose_op = P.Transpose()
        self.norm_op = N.Norm()
        self.normalize_op = P.LayerNorm()
        self.reduce_max_op = P.ReduceMax()
        self.reduce_min_op = P.ReduceMin()
        self.concat_v_op = P.Concat(axis=1)
        self.maximum_op = P.Maximum()
        self.squeeze_op = P.Squeeze()
        self.tanh_op = P.Tanh()
        self.reshape_op = P.Reshape()
        self.argmax_op = P.Argmax()
        self.expand_dims = P.ExpandDims()
        self.reciprocal_op = P.Reciprocal()
        self.fill_op = P.Fill()
        self.sign_op = P.Sign()
        self.select_op = P.Select()

        # 没被使用的算子需要注释掉
        if not IS_EXPORT:
            self.accuracy_op = N.Accuracy('classification')

        self.input_x_shape = (46330, 78)
        self.input_y_shape = (46330, 2)
        # UniformReal在云上导出时报错，只能在init时提前构造好
        self.features_random_weight_list = Tensor(np.random.random((self.N2, self.input_x_shape[1] + 1, self.N1)),
                                                  dtype.float32)
        self.enhance_random_weight = Tensor(np.random.random((self.N2 * self.N1 + 1, self.N3)), dtype.float32)

        self.cascade_feature_weight_list = Tensor(np.random.random((self.N2, self.N1, self.N1)), dtype.float32)


    def construct(self, _train_data, _train_label):
        output, weight, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, \
        parameter_of_shrink = self.train(_train_data, _train_label)
        return output, weight

    def train(self, x, y):
        standardized_train_x = self.standardize_input(x)
        output_of_feature_mapping_layer, weight_of_feature_layer, cascade_feature_weights,  max_list_set, min_list_set = \
            self.generate_mapped_features(standardized_train_x)
        input_of_enhance_layer_with_bias = \
            self.generate_input_of_enhance_layer(output_of_feature_mapping_layer)
        weight_of_enhance_layer = self.generate_random_weight_of_enhance_layer()
        output_of_enhance_layer, parameter_of_shrink = \
            self.generate_output_of_enhance_layer(input_of_enhance_layer_with_bias, weight_of_enhance_layer)
        output, weight = \
            self.generate_final_output(output_of_feature_mapping_layer, output_of_enhance_layer, y)
        return output, weight, weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights, weight_of_enhance_layer, parameter_of_shrink

    def generate_mapped_features(self, standardized_train_x):
        feature_of_input_data_with_bias = self.generate_features_of_input(standardized_train_x)
        output_of_feature_mapping_layer = []
        cascade_feature_weights = self.fill_op(dtype.float32, (self.N2, self.N1, self.N1), 0)
        output_of_each_window_next = []
        weight_of_feature_layer = self.fill_op(dtype.float32, (self.N2, feature_of_input_data_with_bias.shape[1], self.N1), 0)
        max_list_set = self.fill_op(dtype.float32, (self.N2, self.N1), 0)
        min_list_set = self.fill_op(dtype.float32, (self.N2, self.N1), 0)
        for i in range(self.N2):
            # 生成随机权重
            weight_of_each_window = self.generate_random_weight_of_window(standardized_train_x, i)
            # 生成窗口特征
            temp_feature_of_each_window = \
                self.matmul_op(feature_of_input_data_with_bias, weight_of_each_window)
            # 压缩
            feature_of_each_window, _, _ = self.mapminmax(temp_feature_of_each_window, -1.0, 1.0)
            # 通过稀疏化计算，生成最终权重
            beta = self.sparse_bls(feature_of_each_window, feature_of_input_data_with_bias)
            # 计算窗口输出 T1
            output_of_each_window_next, cascade_feature_weights = self.compute_window_output(feature_of_input_data_with_bias, beta, cascade_feature_weights, output_of_each_window_next, i)
            # 压缩
            output_of_each_window_next, max_list, min_list = \
                self.mapminmax(output_of_each_window_next, 0.0, 1.0)
            # 拼接窗口输出
            output_of_feature_mapping_layer = \
                self.concatenate_window_output(output_of_feature_mapping_layer, output_of_each_window_next)
            # 更新输出的权重
            weight_of_feature_layer[i] = beta
            max_list_set[i] = max_list
            min_list_set[i] = min_list

        output_of_feature_mapping_layer = \
            self.window_output_stack_to_tensor(output_of_feature_mapping_layer)
        return output_of_feature_mapping_layer, weight_of_feature_layer, cascade_feature_weights, max_list_set, min_list_set,

    def generate_random_weight_of_enhance_layer(self):
        weight_of_enhance_layer = []
        # rand = self.uniform_rand_op((self.N2 * self.N1 + 1, self.N3))
        rand = self.enhance_random_weight
        weight_of_enhance_layer.append(self.orthonormalize(2 * rand - self.fill_op(dtype.float32, rand.shape, 1.0)))
        return self.stack_v_op(weight_of_enhance_layer)

    def generate_final_output(self, _output_of_feature_mapping_layer, _output_of_enhance_layer, _train_label):

        # 拼接T2和y, 生成T3
        concatenate_of_two_layer = self.concat_v_op((_output_of_feature_mapping_layer, _output_of_enhance_layer))

        _output_weight = self.generate_pseudo_inverse(concatenate_of_two_layer, _train_label)
        # 生成训练输出
        _output_of_train = self.generate_result_of_output_layer(concatenate_of_two_layer, _output_weight)
        return _output_of_train, _output_weight

    def standardize_input(self, _train_data):
        zscore, _, _ = self.normalize_op(_train_data, self.fill_op(dtype.float32, (_train_data.shape[1],), 1),
                                         self.fill_op(dtype.float32, (_train_data.shape[1],), 0))
        return zscore

    def generate_random_weight_of_window(self, standardized_x, i):
        # weight = 2 * self.uniform_rand_op((standardized_x.shape[1] + 1, self.N1)) - 1  # 生成每个窗口的权重系数，最后一行为偏差
        weight = 2.0 * self.features_random_weight_list[i] - 1.0  # 生成每个窗口的权重系数，最后一行为偏差
        return weight

    def generate_features_of_input(self, standardized_train_x):
        ones = self.fill_op(dtype.float32, (standardized_train_x.shape[0], 1), 0.1)
        feature_of_input_data_with_bias = self.concat_v_op((standardized_train_x, ones))
        return feature_of_input_data_with_bias

    def mapminmax(self, matrix, v_min=-1.0, v_max=1.0):
        max_list = self.reduce_max_op(matrix, 0)
        min_list = self.reduce_min_op(matrix, 0)
        max_min_dist = max_list - min_list
        mat_min_dist = matrix - min_list
        std = mat_min_dist / max_min_dist
        temp_scare = std * (v_max - v_min)
        xScale = temp_scare + v_min
        return xScale, max_list, min_list

    def sparse_bls(self, Z, x):
        iters = 50
        lam = 0.001
        m = Z.shape[1]
        n = x.shape[1]
        wk = ok = uk = self.fill_op(dtype.float32, (m, n), 0.0)
        L1 = self.matmul_ta_op(Z, Z) + self.eye_op(m, m, dtype.float32)
        L1_inv = self.pinv_svd(L1)
        L2 = self.matmul_op(self.matmul_tb_op(L1_inv, Z), x)
        for i in range(iters):
            ck = L2 + self.matmul_op(L1_inv, ok - uk)
            ok = self.shrinkage(ck + uk, lam)
            cp = ck - ok
            uk = uk + cp
            wk = ok
        return self.transpose_op(wk, (1, 0))

    def shrinkage(self, a, b):
        output = self.maximum_op(a - b, 0.0) - self.maximum_op(-a - b, 0.0)
        return output

    def compute_window_output(self, feature_of_input_data_with_bias, beta, cascade_feature_weights, output_of_each_window_next, i):
        rand = self.cascade_feature_weight_list[i]
        cascade_feature_weight = self.orthonormalize(2 * rand - self.fill_op(dtype.float32, rand.shape, 1.0))
        cascade_feature_weights[i] = cascade_feature_weight

        if i == 0:
            output_of_each_window = self.matmul_op(feature_of_input_data_with_bias, beta)
            output_of_each_window_next = output_of_each_window
        else:
            output_of_each_window_next = self.matmul_op(output_of_each_window_next, cascade_feature_weight)
        return output_of_each_window_next, cascade_feature_weights

    def concatenate_window_output(self, output_of_feature_mapping_layer, t1):
        output_of_feature_mapping_layer.append(t1)
        return output_of_feature_mapping_layer

    def window_output_stack_to_tensor(self, output_of_feature_mapping_layer):
        res = self.stack_v_op(output_of_feature_mapping_layer)
        res = self.reshape_op(res, (res.shape[0], -1))
        return res

    def generate_input_of_enhance_layer(self, _output_of_feature_mapping_layer):
        data_concat_second = self.fill_op(dtype.float32, (_output_of_feature_mapping_layer.shape[0], 1), 0.1)
        input_of_enhance_layer_with_bias = self.concat_v_op((_output_of_feature_mapping_layer, data_concat_second))
        return input_of_enhance_layer_with_bias

    def generate_output_of_enhance_layer(self, _input_of_enhance_layer_with_bias, _weight_of_enhance_layer):
        res_squeeze_input0 = self.squeeze_op(_input_of_enhance_layer_with_bias)
        res_squeeze_input1 = self.squeeze_op(_weight_of_enhance_layer)
        res_matmul = self.matmul_op(res_squeeze_input0, res_squeeze_input1)
        res_reduce_max = self.reduce_max_op(res_matmul)
        parameter_of_shrink = self.s * self.fill_op(dtype.float32, res_reduce_max.shape, 1.0) / res_reduce_max

        res_tanh = self.tanh_op(res_matmul * parameter_of_shrink)
        return res_tanh, parameter_of_shrink

    def generate_pseudo_inverse(self, _concatenate_of_two_layer, _train_y):
        pseudo_inverse = self.pinv(_concatenate_of_two_layer, self.c)
        new_output_weight = self.matmul_op(pseudo_inverse, _train_y)
        return new_output_weight

    def generate_result_of_output_layer(self, concatenate_of_two_layer, output_weight):
        output_of_result = self.matmul_op(concatenate_of_two_layer, output_weight)
        return output_of_result

    def pinv(self, A, reg):
        AAT = self.matmul_ta_op(A, A)
        AATnp = AAT.asnumpy()
        eye = self.eye_op(A.shape[1], A.shape[1], dtype.float32)
        res = reg * eye
        res = res + AAT
        resnp = res.asnumpy()
        res = self.pinv_svd(res)
        res = self.matmul_tb_op(res, A)
        return res

    def orthonormalize(self, _matrix):
        Q, _ = self.qr_decomposition(_matrix)
        return Q

    def copysign(self, x1, x2):
        x1 = self.expand_dims(x1, 0)
        neg_tensor = -1.0 * x1
        pos_tensor = 1.0 * x1
        judge = x1 * x2 < 0.0
        return self.select_op(judge, neg_tensor, pos_tensor)

    # def copysign(self, x1, x2):
    #     x1_sign = self.sign_op(x1)
    #     x2_sign = self.sign_op(x2[0])
    #     return x1_sign*x2_sign*x1

    def qr_decomposition(self, _A):
        (m, n) = _A.shape
        R = _A
        Q = self.eye_op(m, m, dtype.float32)
        for j in range(n):
            # Apply Householder transformation.
            V = R[:, j]
            if j > 0:
                V[0:j] = 0.0
            V = self.expand_dims(V, 1)
            # copy sign operation
            x1 = self.norm_op(V[j:])
            v = V / (V[j] + self.copysign(x1, V[j]))
            v[j] = 1.0
            tau = 2.0 / self.matmul_ta_op(v, v)
            H = self.eye_op(m, m, dtype.float32)
            H_part = tau * self.matmul_tb_op(v, v)
            H_complete = H_part
            H -= H_complete
            R = self.matmul_op(H, R)
            Q = self.matmul_op(H, Q)
        return self.transpose_op(Q[:n], (1, 0)), R[:n]

    def pinv_svd(self, _A):
        u, s, vt = self.svd(_A)
        s = self.reciprocal_op(s)
        res = self.matmul_op(self.transpose_op(vt, (1, 0)), s * self.transpose_op(u, (1, 0)))
        return res

    def diag(self, _M):
        (n, m) = _M.shape
        s = []
        for i in range(m):
            s.append(_M[i][i])
        sigma = self.stack_h_op(s)
        return sigma

    def svd(self, _A):
        Sigma = 0.0
        U = []
        (a, b) = _A.shape
        V = self.eye_op(b, b, dtype.float32)
        for i in range(10):
            U, _ = self.qr_decomposition(self.matmul_op(_A, V))
            V, S = self.qr_decomposition(self.matmul_ta_op(_A, U))
            # _S = self.diag_part_op(S)
            # Sigma = self.diag_op(_S)
            _S = self.diag(S)
            Sigma = self.expand_dims(_S, 1)
        return U, Sigma, self.transpose_op(V, (1, 0))


IS_EXPORT = False
if __name__ == '__main__':
    # modelArt上面的训练路径和数据路径
    if not IS_EXPORT:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        bin_file_path_x = "../../data/train_x.bin"
        bin_file_path_y = "../../data/train_y.bin"
    else:
        import moxing as mox
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        mox.file.shift('os', 'mox')
        bin_file_path_x = "obs://scut-bls/bls/train_x.bin"
        bin_file_path_y = "obs://scut-bls/bls/train_y.bin"


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_url')
    # parser.add_argument('--train_url')
    # args = parser.parse_args()
    bin_file = open(bin_file_path_x, 'rb')
    data = bin_file.read()
    bin_file.close()
    train_data = Tensor(np.ndarray((46330, 78), float, buffer=data), dtype.float32)
    bin_file = open(bin_file_path_y, 'rb')
    data = bin_file.read()
    bin_file.close()
    train_label = Tensor(np.ndarray((46330, 2), float, buffer=data), dtype.float32)

    inputData = np.ceil(train_data.shape[0] * 0.7)
    train_x = train_data[0:(int)(inputData), :]  # training data at the beginning of the incremental learning
    train_y = train_label[0:(int)(inputData), :]  # training labels at the beginning of the incremental learning

    if not IS_EXPORT:
        i = 0
        while True:
            # print(">>>>>>>>>>>>>>>>>>>>>>")
            bls = CFBLSBasicTrain()
            output_of_train, weight_of_train, weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights, weight_of_enhance_layer, parameter_of_shrink\
                = bls.train(train_data, train_label)
            # print(">>>>>>>>>>>>>>>>>>>>>>")
            bls.accuracy_op.update(output_of_train, bls.argmax_op(train_label))
            print("BLS-Library Training Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
            if bls.accuracy_op.eval() > 0.96:
                print("step", i)
                # save_checkpoint(
                #     [{"name": 'weight', "data": weight_of_train}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
                #      {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
                #      {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
                #      {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
                #     '../../data/cfbls_paramtrain_'+ str(i) + '.ckpt')
                break
            i+=1
    else:
        bls = BLSBasicTrain()
        export(bls, train_data, train_label, file_name="bls.air", file_format='AIR')
        mox.file.rename('bls.air', 'obs://scut-bls/bls/bls.air')