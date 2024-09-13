import numpy
import numpy as np
from mindspore import Tensor, dtype
from mindspore.train.serialization import export, save_checkpoint
import mindspore.dataset as ds
import mindspore.context as context
import mindspore.ops as ops
import mindspore.nn as N
import mindspore.numpy as mnp


class BLSBasicTrain(N.Cell):
    def __init__(self) -> None:
        super(BLSBasicTrain, self).__init__()
        self.s = 0.8
        self.c = 2 ** -15
        self.n1 = 10
        self.n2 = 10
        self.n3 = 20
        self.y_max = 1
        self.y_min = 0
        self.iterations = 2

        # 用于训练结果输出
        self.argmax_op = ops.Argmax()
        self.sign_op = ops.Sign()
        self.select_op = ops.Select()
        self.accuracy_op = N.Accuracy('classification')

    def construct(self, _train_data, _train_label):
        output, weight, _, _, _, _, _ = self.train(_train_data, _train_label)
        return output, weight

    def train(self, x, y):
        standardized_data = self.standardize_input(x)
        feature, mapped_features, _, _ = self.generate_mapped_features(standardized_data)
        feature_with_bias = self.enhance_layer_input(feature)
        enhance_layer_weight = self.generate_random_weight_of_enhance_layer()
        enhance_layer_output, shrink_parameter = self.enhance_layer_output(feature_with_bias, enhance_layer_weight)
        output, output_weight = self.final_output(feature, enhance_layer_output, y)
        return output, output_weight, mapped_features, _, _, enhance_layer_weight, shrink_parameter

    def generate_mapped_features(self, standardized_train_x):
        feature = self.input_features(standardized_train_x)
        output = []
        weight = mnp.full((self.n2, feature.shape[1], self.n1), 0.0)
        max_list = mnp.full((self.n2, self.n1), 0.0)
        min_list = mnp.full((self.n2, self.n1), 0.0)
        for i in range(self.n2):
            # 生成随机权重
            weight_of_each_window = self.generate_random_weight_of_window(standardized_train_x, i)
            # 生成窗口特征
            temp_feature_of_each_window = mnp.matmul(feature, weight_of_each_window)
            # 压缩
            feature_of_each_window, _, _ = self.mapminmax(temp_feature_of_each_window, -1.0, 1.0)
            # 通过稀疏化计算，生成最终权重
            beta = self.sparse_bls(feature_of_each_window, feature)
            # 计算窗口输出 T1
            output_of_each_window_next = self.window_output(feature, beta)
            # 压缩
            output_of_each_window_next, max_list, min_list = self.mapminmax(output_of_each_window_next, 0.0, 1.0)
            # 拼接窗口输出
            output = self.concat_window_output(output, output_of_each_window_next)
            # 更新输出的权重
            weight[i] = beta
            max_list[i] = max_list
            min_list[i] = min_list
        output = self.stack_window_output(output)
        return output, weight, max_list, min_list

    def generate_random_weight_of_enhance_layer(self):
        weight = []
        uniform = ops.UniformReal(seed=2)
        rand = uniform((self.n2 * self.n1 + 1, self.n3), 0.0, 1.0)
        weight.append(self.orthonormalize(2 * rand - mnp.full(rand.shape, 1.0)))
        return mnp.stack(weight, axis=1)

    def final_output(self, _output_of_feature_mapping_layer, _output_of_enhance_layer, _train_label):
        # 拼接T2和y, 生成T3
        concat = mnp.concatenate((_output_of_feature_mapping_layer, _output_of_enhance_layer), axis=1)
        weight = self.pseudo_inverse(concat, _train_label)
        # 生成训练输出
        output = self.output_layer(concat, weight)
        return output, weight

    def generate_random_weight_of_window(self, standardized_x, i):
        uniform = ops.UniformReal(seed=2)
        weight = 2.0 * uniform((standardized_x.shape[1] + 1, self.n1)) - 1.0 # 生成每个窗口的权重系数，最后一行为偏差
        return weight

    def input_features(self, standardized_train_x):
        ones = mnp.full((standardized_train_x.shape[0], 1), 0.1)
        feature_of_input_data_with_bias = mnp.concatenate((standardized_train_x, ones), axis=1)
        return feature_of_input_data_with_bias

    def window_output(self, feature_of_input_data_with_bias, beta):
        output_of_each_window = mnp.matmul(feature_of_input_data_with_bias, beta)
        return output_of_each_window

    def concat_window_output(self, output_of_feature_mapping_layer, t1):
        output_of_feature_mapping_layer.append(t1)
        return output_of_feature_mapping_layer

    def stack_window_output(self, output_of_feature_mapping_layer):
        res = mnp.stack(output_of_feature_mapping_layer, axis=1)
        res = mnp.reshape(res, (res.shape[0], -1))
        return res

    def enhance_layer_input(self, mapped_feature):
        data_concat_second = mnp.full((mapped_feature.shape[0], 1), 0.1)
        res = mnp.concatenate((mapped_feature, data_concat_second), axis=1)
        return res

    def enhance_layer_output(self, _input_of_enhance_layer_with_bias, _weight_of_enhance_layer):
        res_squeeze_input0 = mnp.squeeze(_input_of_enhance_layer_with_bias)
        res_squeeze_input1 = mnp.squeeze(_weight_of_enhance_layer)
        res_matmul = mnp.matmul(res_squeeze_input0, res_squeeze_input1)
        res_reduce_max = mnp.amax(res_matmul)
        shrink_parameter = self.s * mnp.full(res_reduce_max.shape, 1.0) / res_reduce_max
        res_tanh = mnp.tanh(res_matmul * shrink_parameter)
        return res_tanh, shrink_parameter

    def pseudo_inverse(self, _concatenate_of_two_layer, _train_y):
        pseudo_inverse = self.pinv(_concatenate_of_two_layer, self.c)
        new_output_weight = mnp.matmul(pseudo_inverse, _train_y)
        return new_output_weight

    def output_layer(self, concatenate_of_two_layer, output_weight):
        output_of_result = mnp.matmul(concatenate_of_two_layer, output_weight)
        return output_of_result
