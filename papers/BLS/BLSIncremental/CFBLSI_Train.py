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


class BLSIncremental(N.Cell):
    def __init__(self) -> None:
        super(BLSIncremental, self).__init__()
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
        self.features_random_weight_list = Tensor(np.random.random((self.N2, self.input_x_shape[1] + 1, self.N1)),
                                                  dtype.float32)
        self.enhance_random_weight = Tensor(np.random.random((self.N2 * self.N1 + 1, self.m2)), dtype.float32)

    def get_incremental_model_initial_parameter(self, basic_data, parameter):
        "获取增量学习需要的两个中间参数，基础数据集形成的特征层，以及输入层"
        standardized_basic_x = self.standardize_input(basic_data)
        base_feature_mapping_layer = self.generate_new_mapped_features(standardized_basic_x, parameter[1])
        output_of_enhance_layer = \
            self.test_generate_enhance_nodes(base_feature_mapping_layer, parameter[2])
        base_input_of_two_layer = self.concat_v_op((base_feature_mapping_layer, output_of_enhance_layer))
        base_pinv_of_input = self.pinv(base_input_of_two_layer, self.c)
        base_parameter_of_shrinks = []
        [[output_of_train, weight_of_train],
         [weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights],
         [weight_of_enhance_layer, parameter_of_shrink, cascade_enhance_weights]] = parameter
        base_parameter_of_shrinks = parameter_of_shrink
        base_weight_of_enhancement_layer = weight_of_enhance_layer
        incremental_parameter1 = [[weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights], cascade_enhance_weights]

        incremental_parameter2 = [base_feature_mapping_layer, base_weight_of_enhancement_layer, base_parameter_of_shrinks, base_input_of_two_layer, base_pinv_of_input]
        return [incremental_parameter1, incremental_parameter2]

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

    def train(self, x, y, incremental_parameter):
        standardized_train_x = self.standardize_input(x)
        new_output_of_feature_layer = \
            self.generate_new_mapped_features(standardized_train_x, incremental_parameter[0][0])
        new_output_of_enhance_layers, _ = self.generate_new_enhance_node(new_output_of_feature_layer, incremental_parameter[1], incremental_parameter[0][1])
        next_input_of_two_layer, next_output_of_feature_layer, next_output_weight, next_weight_of_enhance_layer, next_parameter_of_shrinks, next_pinv_of_input = \
            self.generate_pinv_weights(new_output_of_feature_layer, new_output_of_enhance_layers,
                                       incremental_parameter[1], y)

        output = self.generate_result_of_output_layer(next_input_of_two_layer, next_output_weight)
        return [[output, next_output_weight], [next_output_of_feature_layer, next_weight_of_enhance_layer, next_parameter_of_shrinks, next_input_of_two_layer, next_pinv_of_input]]


    def generate_new_mapped_features(self, _standardized_test_x, _feature_parameter):
        [weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights]= _feature_parameter
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
        e = len(weights_of_enhance_layer)
        input_of_enhance_layer_with_bias = self.generate_input_of_enhance_layer(output_of_feature_layer)
        for o in range(e):
            # wh = Wh[o]
            weight_of_enhance_layer = self.squeeze_op(weights_of_enhance_layer[o])
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

    def generate_pinv_weights(self, new_output_of_feature_layer, new_output_of_enhance_layer, generate_parameter, y):
        old_output_of_feature_layer, weights_of_enhance_layer, old_parameter_of_shrinks, old_input_of_two_layer, pinv_of_input = generate_parameter
        new_input_of_two_layer = self.concat_v_op((new_output_of_feature_layer, new_output_of_enhance_layer))
        # print(new_input_of_two_layer.shape)
        new_pinv_of_input = self.pinv(new_input_of_two_layer, self.c)
        # print(pinv_of_input.shape)
        # print(new_pinv_of_input.shape)
        next_pinv_of_input = self.concat_v_op((pinv_of_input, new_pinv_of_input))
        # 命名待确认，next_input_of_two_layer1为原输入层+新增数据的特征节点和增强节点
        next_input_of_two_layer1 = self.concat_h_op((old_input_of_two_layer, new_input_of_two_layer))

        next_output_of_feature_layer = self.concat_h_op((old_output_of_feature_layer, new_output_of_feature_layer))

        input_of_enhance_layer_with_bias = self.generate_input_of_enhance_layer(next_output_of_feature_layer)
        output_of_enhance_layer1 = []
        # 生成强化层权重
        weights_of_enhance_layer = self.generate_random_weight_of_enhance_layer(weights_of_enhance_layer)
        # 生成强化层输出
        output_of_enhancement_layer1, parameter_of_shrink = self.generate_output_of_enhance_layer(input_of_enhance_layer_with_bias, weights_of_enhance_layer[-1])
        # next_input_of_two_layer2 就是新的输入层
        next_input_of_two_layer2 = self.concat_v_op((next_input_of_two_layer1, output_of_enhancement_layer1))

        d = self.matmul_op(next_pinv_of_input, output_of_enhancement_layer1)
        c = output_of_enhancement_layer1 - self.matmul_op(next_input_of_two_layer1, d)
        if self.reduce_sum_op(c-0) == 0:
            w = d.shape[1]
            b = np.mat(np.eye(w) + d.T.dot(d)).I.dot(d.T.dot(next_pinv_of_input))
        else:
            b = self.pinv(c, self.c)
        next_pinv_of_input = self.concat_h_op((next_pinv_of_input - self.matmul_op(d, b), b))
        next_output_weight = self.matmul_op(next_pinv_of_input, y)
        next_weights_of_enhance_layer = weights_of_enhance_layer
        old_parameter_of_shrinks.append(parameter_of_shrink)
        return next_input_of_two_layer2, next_output_of_feature_layer, next_output_weight, next_weights_of_enhance_layer, old_parameter_of_shrinks, next_pinv_of_input


    def generate_random_weight_of_enhance_layer(self, weights_of_enhance_layer):
        # weights_of_enhance_layer = []
        rand = self.enhance_random_weight
        weight_of_enhance_layer = self.orthonormalize(2 * rand - self.fill_op(dtype.float32, rand.shape, 1.0))
        weight_of_enhance_layer = self.squeeze_op(weight_of_enhance_layer)
        weights_of_enhance_layer.append(weight_of_enhance_layer)
        return weights_of_enhance_layer


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

    def generate_output_of_enhance_layer(self, _input_of_enhance_layer_with_bias, _weight_of_enhance_layer):
        output_of_enhance_layers = []
        output_of_enhance_layer = []
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

    from BLSAlgorithm.BLSBasic.temp import load_my_checkpoint

    param = load_my_checkpoint('../../data/cfbls_param.ckpt')
    # print(para2["parameter"])
    weight_of_train = param["weight"]
    weight_of_feature_layer = param["weight_of_feature_layer"]
    max_list_set = param["max_list_set"]
    min_list_set = param["min_list_set"]
    weight_of_enhance_layer = param["weight_of_enhance_layer"]
    cascade_feature_weights = param.get("cascade_feature_weights")
    parameter_of_shrink = param["parameter_of_shrink"]
    parameter = [[None, weight_of_train],
                 [weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights],
                 [[weight_of_enhance_layer], [parameter_of_shrink], None]]

    basic_data = train_data[0: (int)(inputData), :]

    if not IS_EXPORT:
        i = 0

        while True:
            # print(">>>>>>>>>>>>>>>>>>>>>>")
            bls = BLSIncremental()
            incremental_param = bls.get_incremental_model_initial_parameter(basic_data, parameter)
            return_parameter = []
            for e in range(0, l - 1):
                while True:
                    print("incremental learning step", e + 1)
                    new_train_x = train_data[(int)(inputData) + e * m: (int)(inputData) + (e + 1) * m, :]
                    new_train_y = train_label[0:(int)(inputData) + (e + 1) * m, :]

                    return_parameter = bls.train(new_train_x, new_train_y, copy.deepcopy(incremental_param))
                    train_result = return_parameter[-+0][0]
                    bls.accuracy_op.update(train_result, bls.argmax_op(new_train_y))
                    print("incremental Trainning Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
                    if bls.accuracy_op.eval() > 0:
                        break
                    # test_result = bls.incremental_testing(test_data, parameter, return_parameter)
                    # # 更新参数
                    # bls.accuracy_op.update(test_result, bls.argmax_op(test_label))
                    # print("incrmental Tesing Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
                incremental_param[1] = return_parameter[1]
            # print(">>>>>>>>>>>>>>>>>>>>>>")
            if bls.accuracy_op.eval() > 0.95:
                save_checkpoint(
                    [{"name": 'weight', "data": weight_of_train}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
                     {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
                     {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
                     {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
                    '../../data/cfblsI_'+ str(i) + '.ckpt')
                break
            i+=1
    else:

        bls = BLSIncremental()
        export(bls, train_data, train_label, file_name="bls.air", file_format='AIR')
        mox.file.rename('bls.air', 'obs://scut-bls/bls/bls.air')