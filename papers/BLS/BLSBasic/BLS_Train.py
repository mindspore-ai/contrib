import struct

import numpy
import numpy as np
from mindspore import Tensor, dtype
from mindspore.train.serialization import export, save_checkpoint
import mindspore.dataset as ds
import mindspore.context as context
import mindspore.ops as P
import mindspore.nn as N
import mindspore.numpy as mnp


class BLSBasicTrain(N.Cell):
    def __init__(self) -> None:
        super(BLSBasicTrain, self).__init__()
        self.s = 0.8
        self.c = 2 ** -15
        self.N1 = 10
        self.N2 = 10
        self.N3 = 20
        self.ymax = 1
        self.ymin = 0
        self.iterations = 2
        self.norm_op = N.Norm()
        self.normalize_op = P.LayerNorm()
        # 没被使用的算子需要注释掉
        self.argmax_op = P.Argmax()
        self.sign_op = P.Sign()
        self.select_op = P.Select()

        # 没被使用的算子需要注释掉
        if not IS_EXPORT:
            self.accuracy_op = N.Accuracy('classification')

        self.input_x_shape = (500, 784)
        self.input_y_shape = (500, 10)

        # UniformReal在云上导出时报错，只能在init时提前构造好
        # 如果UniformReal能用就注释下面的
        # self.uniform_op = P.UniformReal(seed=2)
        # self.features_random_weight_list = self.uniform_op((self.N2, self.input_x_shape[1] + 1, self.N1))

        # 替代方法
        self.features_random_weight_list = Tensor(np.random.random((self.N2, self.input_x_shape[1] + 1, self.N1)), dtype.float32)
        self.enhance_random_weight = Tensor(np.random.random((self.N2 * self.N1 + 1, self.N3)), dtype.float32)

    def construct(self, _train_data, _train_label):
        output, weight, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, \
        parameter_of_shrink = self.train(_train_data, _train_label)
        return output, weight

    def train(self, x, y):
        standardized_train_x = self.standardize_input(x)
        output_of_feature_mapping_layer, weight_of_feature_layer, max_list_set, min_list_set = \
            self.generate_mapped_features(standardized_train_x)
        input_of_enhance_layer_with_bias = \
            self.generate_input_of_enhance_layer(output_of_feature_mapping_layer)
        weight_of_enhance_layer = self.generate_random_weight_of_enhance_layer()
        output_of_enhance_layer, parameter_of_shrink = \
            self.generate_output_of_enhance_layer(input_of_enhance_layer_with_bias, weight_of_enhance_layer)
        output, weight = \
            self.generate_final_output(output_of_feature_mapping_layer, output_of_enhance_layer, y)
        return output, weight, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, parameter_of_shrink

    def generate_mapped_features(self, standardized_train_x):
        feature_of_input_data_with_bias = self.generate_features_of_input(standardized_train_x)
        output_of_feature_mapping_layer = []
        weight_of_feature_layer = mnp.full((self.N2, feature_of_input_data_with_bias.shape[1], self.N1), 0.0)
        max_list_set = mnp.full((self.N2, self.N1), 0.0)
        min_list_set = mnp.full((self.N2, self.N1), 0.0)
        for i in range(self.N2):
            # 生成随机权重
            weight_of_each_window = self.generate_random_weight_of_window(standardized_train_x, i)
            # 生成窗口特征
            temp_feature_of_each_window = \
                mnp.matmul(feature_of_input_data_with_bias, weight_of_each_window)
            # 压缩
            feature_of_each_window, _, _ = self.mapminmax(temp_feature_of_each_window, -1.0, 1.0)
            # 通过稀疏化计算，生成最终权重
            beta = self.sparse_bls(feature_of_each_window, feature_of_input_data_with_bias)
            # 计算窗口输出 T1
            output_of_each_window_next = self.compute_window_output(feature_of_input_data_with_bias, beta)
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
        return output_of_feature_mapping_layer, weight_of_feature_layer, max_list_set, min_list_set

    def generate_random_weight_of_enhance_layer(self):
        weight_of_enhance_layer = []
        rand = self.uniform_op((self.N2 * self.N1 + 1, self.N3))
        # rand = self.enhance_random_weight
        weight_of_enhance_layer.append(self.orthonormalize(2 * rand - mnp.full(rand.shape, 1.0)))
        return mnp.stack(weight_of_enhance_layer, axis=1)

    def generate_final_output(self, _output_of_feature_mapping_layer, _output_of_enhance_layer, _train_label):

        # 拼接T2和y, 生成T3
        concatenate_of_two_layer = mnp.concatenate((_output_of_feature_mapping_layer, _output_of_enhance_layer), axis=1)

        _output_weight = self.generate_pseudo_inverse(concatenate_of_two_layer, _train_label)
        # 生成训练输出
        _output_of_train = self.generate_result_of_output_layer(concatenate_of_two_layer, _output_weight)
        return _output_of_train, _output_weight

    def standardize_input(self, _train_data):
        zscore, _, _ = self.normalize_op(_train_data, mnp.full((_train_data.shape[1],), 1),
                                         mnp.full((_train_data.shape[1],), 0))
        return zscore

    def generate_random_weight_of_window(self, standardized_x, i):
        weight = 2.0 * self.uniform_rand_op((standardized_x.shape[1] + 1, self.N1)) - 1.0  # 生成每个窗口的权重系数，最后一行为偏差
        # weight = 2.0 * self.features_random_weight_list[i] - 1.0  # 生成每个窗口的权重系数，最后一行为偏差
        return weight

    def generate_features_of_input(self, standardized_train_x):
        ones = mnp.full((standardized_train_x.shape[0], 1), 0.1)
        feature_of_input_data_with_bias = mnp.concatenate((standardized_train_x, ones), axis=1)
        return feature_of_input_data_with_bias

    def mapminmax(self, matrix, v_min=-1.0, v_max=1.0):
        # max_list = self.reduce_max_op(matrix, 0)
        max_list = mnp.amax(matrix, axis=0)
        # min_list = self.reduce_min_op(matrix, 0)
        min_list = mnp.amin(matrix, axis=0)
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
        # wk = ok = uk = self.fill_op(dtype.float32, (m, n), 0.0)
        wk = ok = uk = mnp.full((m, n), 0.0)
        # L1 = self.matmul_ta_op(Z, Z) + self.eye_op(m, m, dtype.float32)
        L1 = mnp.matmul(mnp.transpose(Z), Z) + mnp.eye(m, m)
        L1_inv = self.pinv_svd(L1)
        L2 = mnp.matmul(mnp.matmul(L1_inv, mnp.transpose(Z)), x)
        for i in range(iters):
            ck = L2 + mnp.matmul(L1_inv, ok - uk)
            ok = self.shrinkage(ck + uk, lam)
            cp = ck - ok
            uk = uk + cp
            wk = ok
        return mnp.transpose(wk)

    def shrinkage(self, a, b):
        # output = self.maximum_op(a - b, 0.0) - self.maximum_op(-a - b, 0.0)
        output = mnp.maximum(a - b, mnp.array([0.0])) - mnp.maximum(-a - b, mnp.array([0.0]))
        return output

    def compute_window_output(self, feature_of_input_data_with_bias, beta):
        output_of_each_window = mnp.matmul(feature_of_input_data_with_bias, beta)
        return output_of_each_window

    def concatenate_window_output(self, output_of_feature_mapping_layer, t1):
        output_of_feature_mapping_layer.append(t1)
        return output_of_feature_mapping_layer

    def window_output_stack_to_tensor(self, output_of_feature_mapping_layer):
        res = mnp.stack(output_of_feature_mapping_layer, axis=1)
        res = mnp.reshape(res, (res.shape[0], -1))
        return res

    def generate_input_of_enhance_layer(self, _output_of_feature_mapping_layer):
        data_concat_second = mnp.full((_output_of_feature_mapping_layer.shape[0], 1), 0.1)
        input_of_enhance_layer_with_bias = mnp.concatenate((_output_of_feature_mapping_layer, data_concat_second), axis=1)
        return input_of_enhance_layer_with_bias

    def generate_output_of_enhance_layer(self, _input_of_enhance_layer_with_bias, _weight_of_enhance_layer):
        res_squeeze_input0 = mnp.squeeze(_input_of_enhance_layer_with_bias)
        res_squeeze_input1 = mnp.squeeze(_weight_of_enhance_layer)
        res_matmul = mnp.matmul(res_squeeze_input0, res_squeeze_input1)
        res_reduce_max = mnp.amax(res_matmul)
        parameter_of_shrink = self.s * mnp.full(res_reduce_max.shape, 1.0) / res_reduce_max

        res_tanh = mnp.tanh(res_matmul * parameter_of_shrink)
        return res_tanh, parameter_of_shrink

    def generate_pseudo_inverse(self, _concatenate_of_two_layer, _train_y):
        pseudo_inverse = self.pinv(_concatenate_of_two_layer, self.c)
        new_output_weight = mnp.matmul(pseudo_inverse, _train_y)
        return new_output_weight

    def generate_result_of_output_layer(self, concatenate_of_two_layer, output_weight):
        output_of_result = mnp.matmul(concatenate_of_two_layer, output_weight)
        return output_of_result

    def pinv(self, A, reg):
        AAT = mnp.matmul(mnp.transpose(A), A)
        eye = mnp.eye(A.shape[1], A.shape[1])
        res = reg * eye
        res = res + AAT
        res = self.pinv_svd(res)
        res = mnp.matmul(res, mnp.transpose(A))
        return res

    def orthonormalize(self, _matrix):
        Q, _ = self.qr_decomposition(_matrix)
        return Q

    def copysign(self, x1, x2):
        x1 = mnp.expand_dims(x1, 0)
        sing_x1 = self.sign_op(x1)
        sing_x2 = self.sign_op(x2)
        decision = sing_x1 * sing_x2
        print("decision: ", decision)
        return decision * x1
        # return mnp.select(judge, neg_tensor, pos_tensor)

    def qr_decomposition(self, _A):
        (m, n) = _A.shape
        R = _A
        Q = mnp.eye(m, m)
        for j in range(n):
            # Apply Householder transformation.
            V = R[:, j]
            if j > 0:
                V[0:j] = 0.0
            V = mnp.expand_dims(V, axis=1)
            # copy sign operation
            x1 = self.norm_op(V[j:])
            print("x1: ", x1)
            # print("x1: ",x1)
            # v = V / (V[j] + self.copysign(x1, V[j]))
            v = V / (V[j] + self.copysign(x1, V[j]))
            v[j] = 1.0
            tau = 2.0 / mnp.matmul(mnp.transpose(v), v)
            H = mnp.eye(m, m)
            H_part = tau * mnp.matmul(v, mnp.transpose(v))
            H_complete = H_part
            H -= H_complete
            R = mnp.matmul(H, R)
            Q = mnp.matmul(H, Q)
        return mnp.transpose(Q[:n]), R[:n]

    def pinv_svd(self, _A):
        u, s, vt = self.svd(_A)
        s = mnp.reciprocal(s)
        res = mnp.matmul(mnp.transpose(vt), s * mnp.transpose(u))
        return res

    def diag(self, _M):
        (n, m) = _M.shape
        s = []
        for i in range(m):
            s.append(_M[i][i])
        sigma = mnp.stack(s, axis=0)
        return sigma

    def svd(self, _A):
        Sigma = 0.0
        U = []
        (a, b) = _A.shape
        V = mnp.eye(b, b)
        for i in range(3):
            U, _ = self.qr_decomposition(mnp.matmul(_A, V))
            V, S = self.qr_decomposition(mnp.matmul(mnp.transpose(_A), U))
            # _S = self.diag_part_op(S)
            # Sigma = self.diag_op(_S)
            Sigma = mnp.diag(S)
        return U, mnp.expand_dims(Sigma, axis=1), mnp.transpose(V)


DEVICE_TARGET = "Ascend"
IS_EXPORT = True
IS_MODEL_ART = False

if __name__ == '__main__':
    # 训练路径和数据路径

    context.set_context(mode=context.GRAPH_MODE, device_target=DEVICE_TARGET)
    mnist_train = ds.MnistDataset(dataset_dir="../../data/mnist/train")
    mnist_test = ds.MnistDataset(dataset_dir="../../data/mnist/test")
    train_data = []
    train_label = []
    for data in mnist_train.create_dict_iterator():
        train_data.append(data['image'].asnumpy().reshape(784).tolist())
        temp_label = np.zeros(10)
        temp_label[data['label'].asnumpy()] = 1
        train_label.append(temp_label.tolist())
        # print("Image shape:", data['image'].shape, ", Label:", data['label'])
    print("Dataset read complete")
    train_data = Tensor(train_data, dtype.float32)
    train_label = Tensor(train_label, dtype.float32)
    print("Tensor generated, data shape:{}, label shape:{}".format(train_data.shape, train_label.shape))
    # if IS_MODEL_ART:
    #     import moxing as mox
    #     mox.file.shift('os', 'mox')
    #     bin_file_path_x = "obs://scut-bls/bls/train_x.bin"
    #     bin_file_path_y = "obs://scut-bls/bls/train_y.bin"
    # else:
    #     bin_file_path_x = "../../data/train_x.bin"
    #     bin_file_path_y = "../../data/train_y.bin"

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_url')
    # parser.add_argument('--train_url')
    # args = parser.parse_args()
    if not IS_EXPORT:
        print("training1..")
        bls = BLSBasicTrain()
        print("training2..")
        print(">>>>>>>>>>>>>>>>>>>>>>")
        output_of_train, weight_of_train, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, parameter_of_shrink \
            = bls.train(train_data, train_label)
        print(">>>>>>>>>>>>>>>>>>>>>>")
        bls.accuracy_op.update(output_of_train, bls.argmax_op(train_label))
        print("BLS-Library Training Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
        # save_checkpoint(
        #     [{"name": 'weight', "data": weight_of_train}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
        #      {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
        #      {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
        #      {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
        #     './../data/param2.ckpt')
    else:
        print("Starting initialization of BLS instance...")
        bls = BLSBasicTrain()
        print("Initialization complete")
        export(bls, train_data, train_label, file_name="bls.air", file_format='AIR')
        if IS_MODEL_ART:
            # mox.file.rename('bls.air', 'obs://scut-bls/bls/bls.air')
            pass
