"""
Filename: communication_gan.py
Author: fangxiuwen
Contact: fangxiuwen67@163.com
"""
import os
import copy
import mindspore
import mindspore.dataset as ds
from mindspore import nn
from mindspore import Model
from mindspore.dataset import transforms, vision
import mindspore.ops as ops
from data_utils import FemnistValTest

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback

def mkdirs(dirpath):
    """
    Create Folder
    """
    try:
        os.makedirs(dirpath)
    except OSError as _:
        pass

def get_model_list(root, name, models_ini_list, models):
    """
    Load a saved model for testing
    """
    model_list = []
    for i in range(len(name)):
        data_path = os.path.join(root, name[i])
        param_dict = load_checkpoint(data_path)
        net = models[models_ini_list[i]["model_type"]](models_ini_list[i]["params"])
        load_param_into_net(net, param_dict)
        model_list.append(net)
    return model_list


def get_femnist_model_list(root, name, models_ini_list, models):
    """
    Load a saved model for testing
    """
    model_list = []
    for i in range(len(name)):
        if i in [0, 1, 2, 3, 4]:
            param_dict = load_checkpoint(os.path.join(root[i], name[i]))
            net = models[models_ini_list[i]["model_type"]](models_ini_list[i]["params"])
            load_param_into_net(net, param_dict)
            model_list.append(net)
    return model_list

def test_models_femnist(models_list, test_x, test_y):
    """
    Test models
    """
    dataset_sink = mindspore.context.get_context('device_target') == 'CPU'
    apply_transform = transforms.py_transforms.Compose([vision.py_transforms.ToTensor(),
                                                        vision.py_transforms.Normalize((0.1307,), (0.3081,))])
    femnist_bal_data_test = FemnistValTest(test_x, test_y, apply_transform)
    testloader = ds.GeneratorDataset(femnist_bal_data_test, ["data", "label"], shuffle=True)
    testloader = testloader.batch(batch_size=128)

    accuracy_list = []
    loss = NLLLoss()
    for _, model in enumerate(models_list):
        model = Model(model, loss, metrics={"accuracy"})
        acc = model.eval(testloader, dataset_sink_mode=dataset_sink)
        accuracy = acc['accuracy']
        accuracy_list.append(accuracy)
    print(accuracy_list)
    return accuracy_list

def average_weights(w):
    """
    Average model weights
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        div = ops.Div()
        w_avg[key] = div(w_avg[key], len(w))
    return w_avg

class NLLLoss(nn.LossBase):
    """
    NLLLoss loss function
    """
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, logits, label):
        label_one_hot = self.one_hot(label, ops.shape(logits)[-1], ops.scalar_to_array(1.0), ops.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss)

class EarlyStop(Callback):
    """
    Early stopping
    """
    def __init__(self, control_loss=1):
        super(EarlyStop, self).__init__()
        self._control_loss = control_loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if loss.asnumpy() < self._control_loss:
            # Stop training
            run_context._stop_requested = True
