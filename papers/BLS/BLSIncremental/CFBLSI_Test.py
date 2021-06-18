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
from utils.BLS_load_checkpoint import *


class CFBLSIncrementalTest(N.Cell):
    def __init__(self) -> None:
        super(CFBLSIncrementalTest, self).__init__()
        self.s = 0.8
        self.c = 2 ** -15
        self.N1 = 10
        self.N2 = 10
        self.N3 = 20
        self.m2 = 10
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
        # 没被使用的算子需要注释掉
        # self.uniform_rand_op = P.UniformReal()
        self.reduce_max_op = P.ReduceMax()
        self.reduce_min_op = P.ReduceMin()
        self.concat_v_op = P.Concat(axis=1)
        self.concat_h_op = P.Concat(axis=0)
        self.maximum_op = P.Maximum()
        self.squeeze_op = P.Squeeze()
        self.tanh_op = P.Tanh()
        self.reshape_op = P.Reshape()
        self.argmax_op = P.Argmax()
        self.expand_dims = P.ExpandDims()
        self.reciprocal_op = P.Reciprocal()
        self.fill_op = P.Fill()
        self.select_op = P.Select()
        self.reduce_sum_op = P.ReduceSum()

        # 没被使用的算子需要注释掉
        if not IS_EXPORT:
            self.accuracy_op = N.Accuracy('classification')

        self.input_x_shape = (46330, 78)
        self.input_y_shape = (46330, 2)
        # UniformReal在云上导出时报错，只能在init时提前构造好
        self.enhance_random_weight = Tensor(np.random.random((self.N2 * self.N1 + 1, self.m2)), dtype.float32)


    def test_generate_enhance_nodes(self, _output_of_feature_mapping_layer, _enhance_parameter):
        [[weight_of_enhance_layer], [parameter_of_shrink], cascade_enhance_weights] = _enhance_parameter
        input_of_enhance_layer_with_bias = self.generate_input_of_enhance_layer(_output_of_feature_mapping_layer)
        output_of_enhance_layer = self.test_generate_output_of_enhance_layer(input_of_enhance_layer_with_bias,
                                                               weight_of_enhance_layer,
                                                               parameter_of_shrink=parameter_of_shrink)
        return output_of_enhance_layer


    def test_generate_output_of_enhance_layer(self, input_of_enhance_layer_with_bias, weight_of_enhance_layer, parameter_of_shrink):
        res_squeeze_input0 = self.squeeze_op(input_of_enhance_layer_with_bias)
        res_squeeze_input1 = self.squeeze_op(weight_of_enhance_layer)
        res_matmul = self.matmul_op(res_squeeze_input0, res_squeeze_input1)
        res_tanh = self.tanh_op(res_matmul * parameter_of_shrink)
        print(res_tanh.shape)
        return res_tanh

    def test(self, test_data, parameter, return_parameter):
        [[output_of_result, next_output_weight],
         [next_output_of_feature_mapping_layer, next_weight_of_enhancement_layer, parameter_of_shrink, next_input_of_two_layer,
          next_pinv_of_input]] = return_parameter

        [[output_of_train, output_weight],
         [weights_of_feature_layer, max_list_set, min_list_set,cascade_feature_weights],
         [weights_of_enhancement_layer, parameter_of_shrink]] = parameter

        # 增量训练模型
        standardized_test_x = self.standardize_input(test_data)
        # 生成特征层
        output_of_feature_mapping_layer = self.generate_new_mapped_features(standardized_test_x, parameter[1])
        # 生成强化层
        output_of_enhancement_layer = []

        output_of_enhancement_layer, _ = self.generate_new_enhance_node(
            output_of_feature_mapping_layer, return_parameter[1], None)

        test_result = self.generate_final_output(next_output_weight, output_of_feature_mapping_layer, output_of_enhancement_layer)
        return test_result



    def generate_new_mapped_features(self, _standardized_test_x, _feature_parameter):
        [weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights] = _feature_parameter
        feature_of_input_data_with_bias = self.generate_features_of_input(_standardized_test_x)
        output_of_feature_mapping_layer = []
        output_of_each_window_test_next = []
        for i in range(self.N2):
            wha = cascade_feature_weights[i]
            if i == 0:
                output_of_each_window_test = self.matmul_op(feature_of_input_data_with_bias, weight_of_feature_layer[i])
                output_of_each_window_test_next = output_of_each_window_test
            else:
                output_of_each_window_test_next = self.matmul_op(output_of_each_window_test_next, wha)
                output_of_each_window_test = output_of_each_window_test_next

            output_of_each_window_test, _, _ = self.mapminmax(output_of_each_window_test, max_list_set[i], min_list_set[i], 0, 1)
            output_of_feature_mapping_layer = self.concatenate_window_output(output_of_feature_mapping_layer, output_of_each_window_test)
        output_of_feature_mapping_layer = self.window_output_stack_to_tensor(output_of_feature_mapping_layer)
        return output_of_feature_mapping_layer

    def generate_new_enhance_node(self, output_of_feature_layer, parameter, cascade_enhance_weights):
        [_, weights_of_enhance_layer, parameter_of_shrink, _, _] = parameter
        output_of_enhance_layers = []
        output_of_enhance_layer = []
        e = 1 + weights_of_enhance_layer[1].shape[0]
        input_of_enhance_layer_with_bias = self.generate_input_of_enhance_layer(output_of_feature_layer)
        for o in range(e):
            # wh = Wh[o]
            if o == 0:
                weight_of_enhance_layer = weights_of_enhance_layer[0]
            else:
                weight_of_enhance_layer = weights_of_enhance_layer[1][o-1]
            temp_of_output_of_enhance_layer = self.matmul_op(input_of_enhance_layer_with_bias, weight_of_enhance_layer)
            temp_of_output_of_enhance_layer = self.squeeze_op(temp_of_output_of_enhance_layer)
            output_of_enhance_layer = temp_of_output_of_enhance_layer * parameter_of_shrink[o]
            output_of_enhance_layer = self.tanh_op(output_of_enhance_layer)
            output_of_enhance_layer = self.squeeze_op(output_of_enhance_layer)
            if o == 0:
                output_of_enhance_layers = output_of_enhance_layer
            else:
                output_of_enhance_layers = self.concat_v_op((output_of_enhance_layers, output_of_enhance_layer))
        return output_of_enhance_layers, output_of_enhance_layer





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


    def generate_final_output(self, _output_weight, _output_of_feature_mapping_layer, _output_of_enhance_layer):
        concatenate_of_two_layer = self.concat_v_op((_output_of_feature_mapping_layer, _output_of_enhance_layer))
        output_of_test = self.matmul_op(concatenate_of_two_layer, _output_weight)

        return output_of_test

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
        for i in range(3):
            U, _ = self.qr_decomposition(self.matmul_op(_A, V))
            V, S = self.qr_decomposition(self.matmul_ta_op(_A, U))
            # _S = self.diag_part_op(S)
            # Sigma = self.diag_op(_S)
            Sigma = self.diag(S)
        return U, self.expand_dims(Sigma,1), self.transpose_op(V, (1, 0))


IS_EXPORT = False
if __name__ == '__main__':
    # modelArt上面的训练路径和数据路径
    if not IS_EXPORT:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        train_file_path_x = "../../data/train_x.bin"
        train_file_path_y = "../../data/train_y.bin"
        test_file_path_x = "../../data/test_x.bin"
        test_file_path_y = "../../data/test_y.bin"

    else:
        import moxing as mox
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        mox.file.shift('os', 'mox')
        train_file_path_x = "obs://scut-bls/bls/train_x.bin"
        train_file_path_y = "obs://scut-bls/bls/train_y.bin"
        test_file_path_x = "obs://scut-bls/bls/test_x.bin"
        test_file_path_y = "obs://scut-bls/bls/test_y.bin"



    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_url')
    # parser.add_argument('--train_url')
    # args = parser.parse_args()
    bin_file = open(train_file_path_x, 'rb')
    data = bin_file.read()
    bin_file.close()
    train_data = Tensor(np.ndarray((46330, 78), float, buffer=data), dtype.float32)
    bin_file = open(train_file_path_x, 'rb')
    data = bin_file.read()
    bin_file.close()
    train_label = Tensor(np.ndarray((46330, 2), float, buffer=data), dtype.float32)

    bin_file = open(test_file_path_x, 'rb')
    data = bin_file.read()
    bin_file.close()
    test_data = Tensor(np.ndarray((86534, 78), float, buffer=data), dtype.float32)
    bin_file = open(test_file_path_y, 'rb')
    data = bin_file.read()
    bin_file.close()
    test_label = Tensor(np.ndarray((86534, 2), float, buffer=data), dtype.float32)

    C = 2 ** -25  # parameter for sparse regularization
    s = 0.8  # the shrinkage parameter for enhancement nodes
    l = 5
    m2 = 10
    N1_bls = 10
    N2_bls = 10
    N3_bls = 20
    inputData = np.ceil(train_data.shape[0] * 0.7)
    m = int(np.ceil((train_data.shape[0] - inputData) / l))

    parameter = load_cfbls_basic_ckpt('../../checkpoints/cfParam.ckpt')
    basic_data = train_data[0: (int)(inputData), :]

    if not IS_EXPORT:
        # print(">>>>>>>>>>>>>>>>>>>>>>")
        bls = CFBLSIncrementalTest()
        for e in range(0, l - 1):
            return_parameter = load_bls_incremental_ckpt("../../checkpoints/increCFParam_"+str(e)+".ckpt")

            print("incremental learning step", e + 1)
            test_result = bls.test(test_data, copy.deepcopy(parameter), copy.deepcopy(return_parameter))
            bls.accuracy_op.update(test_result, bls.argmax_op(test_label))
            print("incremental Trainning Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
            bls.accuracy_op.clear()

    else:

        bls = CFBLSIncrementalTest()
        export(bls, train_data, train_label, file_name="bls.air", file_format='AIR')
        mox.file.rename('bls.air', 'obs://scut-bls/bls/bls.air')