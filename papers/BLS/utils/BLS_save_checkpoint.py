from mindspore import save_checkpoint
from mindspore import Tensor, dtype
from mindspore.train.serialization import export, save_checkpoint
import mindspore.context as context
import mindspore.ops.operations as P
import mindspore.nn as N
import numpy as np

def save_bls_basic_ckpt(param, path1="../../checkpoints/ms_bls_basic.ckpt", path2 = "../../checkpoints/ms_bls_incre_prepare.ckpt"):
    [[output, weight, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, parameter_of_shrink],
     [output_of_feature_mapping_layer, concatenate_of_two_layer, pseudo_inverse]]\
        = param

    save_checkpoint(
        [{"name": 'weight', "data": weight}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
         {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
         {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
         {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
        path1)

    save_checkpoint(
        [{"name": 'feature_layer', "data": output_of_feature_mapping_layer},
         {"name": 'input_layer', "data": concatenate_of_two_layer},
         {"name": 'pseudo_inverse', "data": pseudo_inverse}],
        path2)



def save_bls_incremental_ckpt(incremental_param, path="../../checkpoints/ms_bls_incremental.ckpt"):
    [ [output_of_train, output_weight], [base_feature_mapping_layer, base_weight_of_enhancement_layer, base_parameter_of_shrinks, base_input_of_two_layer,
     base_pinv_of_input]] = incremental_param

    fill_op = P.Fill
    m, n = base_weight_of_enhancement_layer[1].shape
    loop_enhance_weight1 = base_weight_of_enhancement_layer[0]
    loop_enhance_weight2 = fill_op(dtype.float32, (len(base_weight_of_enhancement_layer)-1, m, n), 0)

    save_checkpoint([{"name": 'loop_output_weight', "data": output_weight},
                     {"name": 'loop_feature_weight', "data": base_feature_mapping_layer},
                     {"name": 'loop_enhance_weight1', "data": loop_enhance_weight1},
                     {"name": 'loop_enhance_weight2', "data": loop_enhance_weight2},
                     {"name": 'loop_shrink', "data": base_parameter_of_shrinks},
                     {"name": 'loop_input_layer', "data": base_input_of_two_layer},
                     {"name": 'loop_pinv', "data": base_pinv_of_input}],
                    path
                    )


def save_cfbls_basic_ckpt(param, path="../../checkpoints/ms_bls_basic.ckpt"):
    output, weight, weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights, weight_of_enhance_layer, parameter_of_shrink\
        = param
    save_checkpoint(
        [{"name": 'weight', "data": weight}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
         {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
         {"name": 'cascade_feature_weights', "data": cascade_feature_weights},
         {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
         {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
        path)

def save_cebls_basic_ckpt(param, path1="../../checkpoints/ms_cebls_basic.ckpt", path2 = "../../checkpoints/ms_cebls_incre_prepare.ckpt"):
    [[output, weight, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, cascade_weight_of_enhance_layer, parameter_of_shrink],
     [output_of_feature_mapping_layer, concatenate_of_two_layer, pseudo_inverse]]\
        = param

    save_checkpoint(
        [{"name": 'weight', "data": weight}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
         {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
         {"name": 'cascade_weight_of_enhance_layer', "data": cascade_weight_of_enhance_layer},
         {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
         {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
        path1)

    save_checkpoint(
        [{"name": 'feature_layer', "data": output_of_feature_mapping_layer},
         {"name": 'input_layer', "data": concatenate_of_two_layer},
         {"name": 'pseudo_inverse', "data": pseudo_inverse}],
        path2)




