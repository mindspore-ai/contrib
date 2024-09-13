import mindspore.context as context
import mindspore.dataset as ds
import numpy as np
from mindspore import Tensor, dtype
from mindspore.train.serialization import export, save_checkpoint
from contrib.papers.BLS.BLSBasic.BLS import BLSBasicTrain

DEVICE_TARGET = "Ascend"
IS_EXPORT = True
IS_MODEL_ART = False

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=DEVICE_TARGET)
    mnist_train = ds.MnistDataset(dataset_dir="../../../dataset/mnist/train")
    train_data = []
    train_label = []
    for data in mnist_train.create_dict_iterator():
        image = np.divide(data['image'].asnumpy().reshape(784), 255.0)
        train_data.append(image)
        label = np.asarray([1.0 if x == data['label'].asnumpy() else 0.0 for x in range(10)])
        train_label.append(label)
    print("Dataset read complete")
    train_data = Tensor(train_data, dtype.float32)
    train_label = Tensor(train_label, dtype.float32)
    print("Tensor generated, data shape:{}, label shape:{}".format(train_data.shape, train_label.shape))

    if not IS_EXPORT:
        print("Starting initialization of BLS instance...")
        bls = BLSBasicTrain()
        print("Initialization complete")
        output_of_train, weight_of_train, weight_of_feature_layer, max_list_set, min_list_set, weight_of_enhance_layer, parameter_of_shrink \
            = bls.train(train_data, train_label)
        bls.accuracy_op.update(output_of_train, bls.argmax_op(train_label))
        print("BLS-Library Training Accuracy is : ", bls.accuracy_op.eval() * 100, " %")
        save_checkpoint(
            [{"name": 'weight', "data": weight_of_train}, {"name": 'weight_of_feature_layer', "data": weight_of_feature_layer},
             {"name": 'max_list_set', "data": max_list_set}, {"name": 'min_list_set', "data": min_list_set},
             {"name": 'weight_of_enhance_layer', "data": weight_of_enhance_layer},
             {"name": 'parameter_of_shrink', "data": parameter_of_shrink}],
            '../data/train_param.ckpt')
    else:
        print("Starting initialization of BLS instance...")
        bls = BLSBasicTrain()
        print("Initialization complete")
        export(bls, train_data, train_label, file_name="bls.air", file_format='AIR')
        if IS_MODEL_ART:
            pass
