from mindspore.train.serialization import save_checkpoint
import mindspore as ms
import os
import numpy as np
from mindspore import log as logger
from mindspore.train.checkpoint_pb2 import Checkpoint
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype



def load_bls_basic_ckpt(path1="./checkpoints/ms_bls_basic.ckpt", path2=None):
    param = load_my_checkpoint(path1)
    weight_of_train = param["weight"]
    weight_of_feature_layer = param["weight_of_feature_layer"]
    max_list_set = param["max_list_set"]
    min_list_set = param["min_list_set"]
    weight_of_enhance_layer = param["weight_of_enhance_layer"]
    parameter_of_shrink = param["parameter_of_shrink"]
    if path2 is None:
        parameter = [[None, weight_of_train],
                 [weight_of_feature_layer, max_list_set, min_list_set],
                 [[weight_of_enhance_layer], [parameter_of_shrink]]]
        return parameter

    if path2 is not None:
        param2 = load_my_checkpoint(path2)
        base_input_of_two_layer = param2["input_layer"]
        base_pinv_of_input = param2["pseudo_inverse"]
        base_feature_mapping_layer = param2["feature_layer"]
        base_parameter_of_shrinks = []
        base_parameter_of_shrinks = parameter_of_shrink
        base_weight_of_enhancement_layer = weight_of_enhance_layer
        incremental_parameter1 = [weight_of_feature_layer, max_list_set, min_list_set]

        incremental_parameter2 = [base_feature_mapping_layer, [base_weight_of_enhancement_layer],
                                  [base_parameter_of_shrinks], base_input_of_two_layer, base_pinv_of_input]
        return [incremental_parameter1, incremental_parameter2]

def load_bls_incremental_ckpt(path1="./checkpoints/ms_bls_basic.ckpt", path2 = "./checkpoints/ms_bls_incre_prepare.ckpt"):
    parameter = load_my_checkpoint(path1)
    loop_output_weight = parameter["loop_output_weight"]
    loop_feature_weight = parameter["loop_feature_weight"]
    loop_enhance_weight1 = parameter["loop_enhance_weight1"]
    loop_enhance_weight2 = parameter["loop_enhance_weight2"]
    loop_shrink = parameter["loop_shrink"]
    loop_input_layer = parameter["loop_input_layer"]
    loop_pinv = parameter["loop_pinv"]

    return_parameter = [[None, loop_output_weight],
                        [loop_feature_weight, [loop_enhance_weight1, loop_enhance_weight2], loop_shrink,
                         loop_input_layer, loop_pinv]]
    return return_parameter

def load_cfbls_basic_ckpt(path="./checkpoints/ms_bls_basic.ckpt"):
    param = load_my_checkpoint(path)
    weight_of_train = param["weight"]
    weight_of_feature_layer = param["weight_of_feature_layer"]
    max_list_set = param["max_list_set"]
    min_list_set = param["min_list_set"]
    cascade_feature_weights = param["cascade_feature_weights"]
    weight_of_enhance_layer = param["weight_of_enhance_layer"]
    parameter_of_shrink = param["parameter_of_shrink"]
    parameter = None
    parameter = [[None, weight_of_train],
             [weight_of_feature_layer, max_list_set, min_list_set, cascade_feature_weights],
             [[weight_of_enhance_layer], [parameter_of_shrink]]]
    return parameter

def load_cfbls_incremental_ckpt(path="./checkpoints/ms_bls_basic.ckpt"):
    parameter = load_my_checkpoint(path)
    loop_output_weight = parameter["loop_output_weight"]
    loop_feature_weight = parameter["loop_feature_weight"]
    loop_enhance_weight1 = parameter["loop_enhance_weight1"]
    loop_enhance_weight2 = parameter["loop_enhance_weight2"]
    loop_shrink = parameter["loop_shrink"]
    loop_input_layer = parameter["loop_input_layer"]
    loop_pinv = parameter["loop_pinv"]
    return_parameter = [[None, loop_output_weight],
                        [loop_feature_weight, [loop_enhance_weight1, loop_enhance_weight2], loop_shrink,
                         loop_input_layer, loop_pinv]]
    return return_parameter

def load_cebls_basic_ckpt(path="./checkpoints/ms_bls_basic.ckpt"):
    param = load_my_checkpoint(path)
    weight_of_train = param["weight"]
    weight_of_feature_layer = param["weight_of_feature_layer"]
    max_list_set = param["max_list_set"]
    min_list_set = param["min_list_set"]
    cascade_weight_of_enhance_layer = param["cascade_weight_of_enhance_layer"]
    weight_of_enhance_layer = param["weight_of_enhance_layer"]
    parameter_of_shrink = param["parameter_of_shrink"]
    parameter = None
    parameter = [[None, weight_of_train],
             [weight_of_feature_layer, max_list_set, min_list_set],
             [[weight_of_enhance_layer], [parameter_of_shrink], cascade_weight_of_enhance_layer]]
    return parameter

def load_my_checkpoint(ckpt_file_name):
    """
    this api is copied from mindspore\train\serialization.py, and be modified to be used
    """
    tensor_to_ms_type = {"Int8": mstype.int8, "Uint8": mstype.uint8, "Int16": mstype.int16, "Uint16": mstype.uint16,
                         "Int32": mstype.int32, "Uint32": mstype.uint32, "Int64": mstype.int64, "Uint64": mstype.uint64,
                         "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                         "Bool": mstype.bool_}

    tensor_to_np_type = {"Int8": np.int8, "Uint8": np.uint8, "Int16": np.int16, "Uint16": np.uint16,
                         "Int32": np.int32, "Uint32": np.uint32, "Int64": np.int64, "Uint64": np.uint64,
                         "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_}

    if not isinstance(ckpt_file_name, str):
        raise ValueError("The ckpt_file_name must be string.")

    if not os.path.exists(ckpt_file_name):
        raise ValueError("The checkpoint file is not exist.")

    if ckpt_file_name[-5:] != ".ckpt":
        raise ValueError("Please input the correct checkpoint file name.")

    if os.path.getsize(ckpt_file_name) == 0:
        raise ValueError("The checkpoint file may be empty, please make sure enter the correct file name.")


    logger.info("Execute the process of loading checkpoint files.")
    checkpoint_list = Checkpoint()

    try:
        with open(ckpt_file_name, "rb") as f:
            pb_content = f.read()
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:
        logger.error("Failed to read the checkpoint file `%s`, please check the correct of the file.", ckpt_file_name)
        raise ValueError(e.__str__())

    parameter_dict = {}
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type[data_type]
            ms_type = tensor_to_ms_type[data_type]
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                param_data = np.concatenate((param_data_list), axis=0)
                param_data_list.clear()
                dims = element.tensor.dims

                if dims == [0]:
                    if 'Float' in data_type:
                        param_data = float(param_data[0])
                    elif 'Int' in data_type:
                        param_data = int(param_data[0])
                    parameter_dict[element.tag] = Tensor(param_data, ms_type)
                elif dims == [1]:
                    parameter_dict[element.tag] = Tensor(param_data, ms_type)
                else:
                    param_dim = []
                    for dim in dims:
                        param_dim.append(dim)
                    param_value = param_data.reshape(param_dim)
                    parameter_dict[element.tag] = Tensor(param_value, ms_type)

        logger.info("Loading checkpoint files process is finished.")

    except BaseException as e:
        logger.error("Failed to load the checkpoint file `%s`.", ckpt_file_name)
        raise RuntimeError(e.__str__())

    return parameter_dict