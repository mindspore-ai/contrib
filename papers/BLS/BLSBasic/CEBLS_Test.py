import argparse
import os
# import moxing as mox
from mindspore import Tensor, dtype
from mindspore.train.serialization import export, save_checkpoint
import mindspore.context as context
import mindspore.ops.operations as P
import mindspore.nn as N
import numpy as np
import copy
from utils.BLS_load_checkpoint import *



class CEBLSBasicTest(N.Cell):
    def __init__(self) -> None:
        super(CEBLSBasicTest, self).__init__()
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
        self.eye_op = P.Eye()
        self.transpose_op = P.Transpose()
        self.norm_op = N.Norm()
        self.normalize_op = P.LayerNorm()
        # 没被使用的算子需要注释掉
        self.uniform_rand_op = P.UniformReal()
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
        self.select_op = P.Select()
        self.squeeze_op = P.Squeeze()

        # 没被使用的算子需要注释掉
        self.accuracy_op = N.Accuracy('classification')

        self.input_x_shape = (46330, 78)
        self.input_y_shape = (46330, 2)
        # UniformReal在云上导出时报错，只能在init时提前构造好
        self.features_random_weight_list = Tensor(np.random.random((self.N2, self.input_x_shape[1] + 1, self.N1)),
                                                  dtype.float32)
        self.enhance_random_weight = Tensor(np.random.random((self.N2 * self.N1 + 1, self.N3)), dtype.float32)

    def construct(self, _train_data, _train_label):
        # output, weight, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, parameter_of_shrink = self.train(
        #     _train_data, _train_label)
        # return output, weight
        return self.norm_op(_train_data)

    def test(self, x, parameter):
        standardized_test_x = self.standardize_input(x)
        print(">>>>>>>>>>>>>>>>>>>>>>1")
        output_of_feature_mapping_layer = self.generate_mapped_features(standardized_test_x, parameter[1])
        print(">>>>>>>>>>>>>>>>>>>>>>1")
        output_of_enhance_layer = self.generate_enhance_nodes(output_of_feature_mapping_layer, parameter[2])
        print(">>>>>>>>>>>>>>>>>>>>>>1")
        output_of_test = self.generate_final_output(parameter[0][1], output_of_feature_mapping_layer,
                                                         output_of_enhance_layer)
        return output_of_test

    def generate_mapped_features(self, _standardized_test_x, _feature_parameter):
        [weight_of_feature_layer, max_list_set, min_list_set] = _feature_parameter
        feature_of_input_data_with_bias = self.generate_features_of_input(_standardized_test_x)
        output_of_feature_mapping_layer = []
        output_of_each_window_test_next = []
        for i in range(self.N2):
            weight = weight_of_feature_layer[i]
            output_of_each_window_test = self.matmul_op(feature_of_input_data_with_bias, weight)

            output_of_each_window_test, _, _ = self.mapminmax(output_of_each_window_test, max_list_set[i], min_list_set[i], 0, 1)
            output_of_feature_mapping_layer = self.concatenate_window_output(output_of_feature_mapping_layer, output_of_each_window_test)
        output_of_feature_mapping_layer = self.window_output_stack_to_tensor(output_of_feature_mapping_layer)
        return output_of_feature_mapping_layer

    def generate_enhance_nodes(self, _output_of_feature_mapping_layer, _enhance_parameter):
        [[weight_of_enhance_layer], [parameter_of_shrink], cascade_weight_of_enhance_layer] = _enhance_parameter
        input_of_enhance_layer_with_bias = self.generate_input_of_enhance_layer(_output_of_feature_mapping_layer)
        output_of_enhance_layer = self.generate_output_of_enhance_layer(input_of_enhance_layer_with_bias,
                                                               weight_of_enhance_layer,
                                                               cascade_weight_of_enhance_layer,
                                                               parameter_of_shrink)
        return output_of_enhance_layer

    def generate_input_of_enhance_layer(self, _output_of_feature_mapping_layer):
        data_concat_second = self.fill_op(dtype.float32, (_output_of_feature_mapping_layer.shape[0], 1), 0.1)
        input_of_enhance_layer_with_bias = self.concat_v_op((_output_of_feature_mapping_layer, data_concat_second))
        return input_of_enhance_layer_with_bias

    def generate_output_of_enhance_layer(self, _input_of_enhance_layer_with_bias, _weight_of_enhance_layer, _cascade_weight_of_enhance_layer, _parameter_of_shrink):
        res_squeeze_input0 = self.squeeze_op(_input_of_enhance_layer_with_bias)
        res_squeeze_input1 = self.squeeze_op(_weight_of_enhance_layer)
        res_matmul = []
        for i in range(self.N3):
            if i == 0:
                res_matmul = self.tanh_op(self.matmul_op(res_squeeze_input0, res_squeeze_input1))
            else:
                weight = _cascade_weight_of_enhance_layer[i]

                res_matmul = self.tanh_op(self.matmul_op(res_matmul, weight))
        res_matmul = self.matmul_op(res_squeeze_input0, res_squeeze_input1)
        res_tanh = self.tanh_op(res_matmul * _parameter_of_shrink)
        print(res_tanh.shape)
        return res_tanh

    def generate_final_output(self, _output_weight, _output_of_feature_mapping_layer, _output_of_enhance_layer):
        concatenate_of_two_layer = self.concat_v_op((_output_of_feature_mapping_layer, _output_of_enhance_layer))
        output_of_test = self.matmul_op(concatenate_of_two_layer, _output_weight)

        return output_of_test

    def standardize_input(self, _train_data):
        zscore, _, _ = self.normalize_op(_train_data, self.fill_op(dtype.float32, (_train_data.shape[1],), 1),
                                         self.fill_op(dtype.float32, (_train_data.shape[1],), 0))
        return zscore

    def generate_features_of_input(self, standardized_train_x):
        ones = self.fill_op(dtype.float32, (standardized_train_x.shape[0], 1), 0.1)
        feature_of_input_data_with_bias = self.concat_v_op((standardized_train_x, ones))
        return feature_of_input_data_with_bias

    def mapminmax(self, matrix, max_list, min_list, v_min=-1, v_max=1):
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
        wk = ok = uk = self.fill_op(dtype.float32, (m, n), 0)
        L1 = self.matmul_ta_op(Z, Z) + self.eye_op(m, m, dtype.float32)
        L1_inv = self.pinv_svd(L1)
        L2 = self.matmul_op(self.matmul_tb_op(L1_inv, Z), x)
        for i in range(iters):
            ck = L2 + self.matmul_op(L1_inv, (ok - uk))
            ok = self.shrinkage(ck + uk, lam)
            cp = ck - ok
            uk = uk + cp
            wk = ok
        return self.transpose_op(wk, (1, 0))

    def shrinkage(self, a, b):
        output = self.maximum_op(a - b, 0) - self.maximum_op(-a - b, 0)
        return output

    def compute_window_output(self, feature_of_input_data_with_bias, beta):
        output_of_each_window = self.matmul_op(feature_of_input_data_with_bias, beta)
        return output_of_each_window

    def concatenate_window_output(self, output_of_feature_mapping_layer, t1):
        output_of_feature_mapping_layer.append(t1)
        return output_of_feature_mapping_layer

    def window_output_stack_to_tensor(self, output_of_feature_mapping_layer):
        res = self.stack_v_op(output_of_feature_mapping_layer)
        res = self.reshape_op(res, (res.shape[0], -1))
        return res




    def pinv(self, A, reg):
        AAT = self.matmul_ta_op(A, A)
        eye = self.eye_op(A.shape[1], A.shape[1], dtype.float32)
        res = reg * eye
        res = res + AAT
        res = self.pinv_svd(res)
        res = self.matmul_tb_op(res, A)
        return res

    def orthonormalize(self, _matrix):
        Q, _ = self.qr_decomposition(_matrix)
        return Q

    def copysign(self, x1, x2):
        x1 = self.expand_dims(x1, 0)
        neg_tensor = -1 * x1
        pos_tensor = 1 * x1
        judge = x1 * x2 < 0
        return self.select_op(judge, neg_tensor, pos_tensor)

    def qr_decomposition(self, _A):
        (m, n) = _A.shape
        R = _A
        Q = self.eye_op(m, m, dtype.float32)
        for j in range(n):
            # Apply Householder transformation.
            V = R[j:, j]
            V = self.expand_dims(V, 1)
            # copy sign operation
            x1 = self.norm_op(V)
            v = V / (V[0] + self.copysign(x1, V[0]))
            v[0] = 1
            tau = 2 / self.matmul_ta_op(v, v)
            H = self.eye_op(m, m, dtype.float32)
            H_part = tau * self.matmul_tb_op(v, v)
            op_pad_h = P.Pad(((j, 0), (j, 0)))
            H_complete = op_pad_h(H_part)
            H -= H_complete
            R = self.matmul_op(H, R)
            Q = self.matmul_op(H, Q)
        return self.transpose_op(Q[:n], (1, 0)), R[:n]

    def pinv_svd(self, _A):
        u, s, vt = self.svd(_A)
        s = self.reciprocal_op(s)
        s = self.expand_dims(s, 1)
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
        Sigma = 0
        U = []
        (a, b) = _A.shape
        V = self.eye_op(b, b, dtype.float32)
        for i in range(3):
            U, _ = self.qr_decomposition(self.matmul_op(_A, V))
            V, S = self.qr_decomposition(self.matmul_ta_op(_A, U))
            Sigma = self.diag(S)
        return U, Sigma, self.transpose_op(V, (1, 0))
    def test_pinv(self):
        a = np.random.rand(400,400)
        atensor = Tensor(a, dtype.float32)
        apinv1 = self.pinv_svd(atensor).asnumpy()
        anp1 = np.linalg.pinv(a)
        apinv = self.pinv(atensor, self.c).asnumpy()
        new_output_weight = np.dot(np.linalg.pinv(
            np.dot(a.transpose(), a) + np.identity(
                a.transpose().shape[0]) * self.c), a.transpose())
        print("******")

os.environ['GLOG_v'] = '1'
IS_EXPORT = False
if __name__ == '__main__':

    # modelArt上面的训练路径和数据路径
    if not IS_EXPORT:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        bin_file_path_x = "../../data/test_x.bin"
        bin_file_path_y = "../../data/test_y.bin"
    else:
        import moxing as mox
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        mox.file.shift('os', 'mox')
        bin_file_path_x = "obs://scut-bls/bls/test_x.bin"
        bin_file_path_y = "obs://scut-bls/bls/test_y.bin"


    bin_file = open(bin_file_path_x, 'rb')
    data = bin_file.read()
    bin_file.close()
    test_data = Tensor(np.ndarray((86534, 78), float, buffer=data), dtype.float32)
    bin_file = open(bin_file_path_y, 'rb')
    data = bin_file.read()
    bin_file.close()
    test_label = Tensor(np.ndarray((86534, 2), float, buffer=data), dtype.float32)
    parameter = load_cebls_basic_ckpt('../../checkpoints/ms_cebls_basic.ckpt')
    # 放弃读取文件，构造随机的数据用以导出模型，只需要保证shape与文件中的一样
    # train_data = Tensor(np.random.random((46330, 78)), dtype.float32)
    # train_label = Tensor(np.random.random((46330, 2)), dtype.float32)
    if not IS_EXPORT:
        bls = CEBLSBasicTest()
        # bls.test_pinv()
        print(">>>>>>>>>>>>>>>>>>>>>>")

        output_of_test = bls.test(test_data, parameter)
        print(">>>>>>>>>>>>>>>>>>>>>>")
        bls.accuracy_op.update(output_of_test, bls.argmax_op(test_label))
        print("BLS-Library Testing Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
    else:
        bls = BLSBasicTest()
        export(bls, test_data, parameter, file_name="testbls.air", file_format='AIR')