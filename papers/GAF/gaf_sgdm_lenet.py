"""
Implementation of GAF on LeNet
"""

import os
import mindspore.nn as nn
from mindspore.common.initializer import Normal
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import context
from mindspore.ops import functional as F, composite, operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
import mindspore.ops as ops
from mindspore.ops import MultitypeFuncGraph, Map
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

def create_dataset(data_path, batch_size=32,
                   num_parallel_workers=1):
    """
    Define the dataloader
    """
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)

    return mnist_ds


class LeNet5(nn.Cell):
    """
    Lenet网络
    """

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        """
        Lenet结构
        """
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

_momentum_opt = composite.MultitypeFuncGraph("momentum_opt")

MS_DEV_ENABLE_FALLBACK = 0

@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment, ps_parameter, cache_enable):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        ps_pull_ = P.Pull()
        ps_push_ = P.Push("ApplyMomentum", [])
        shapes = (op_shape(learning_rate), op_shape(gradient), op_shape(momentum))
        success = F.depend(True, ps_pull_(ps_push_((learning_rate, gradient, momentum), shapes), weight))
    else:
        success = F.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    return success

atan = MultitypeFuncGraph('atan')


@atan.register("Tensor")
def tan_tensor(x):
    return 0.05 * ops.atan(20 * x)


atan_gaf = Map()

class Momentum(Optimizer):
    """
    Redefine our optimizer
    """

    @opt_init_args_register
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(Momentum, self).__init__(learning_rate, params, weight_decay, loss_scale)
        Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("For 'Momentum', the argument 'momentum' should be at least 0.0, "
                             "but got {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.params = self.parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)

    def construct(self, gradients):
        """
        working
        """
        params = self.params
        moments = self.moments
        gradients = atan_gaf(atan, gradients)
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)

        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, self.momentum),
                                             lr, gradients, params, moments, self.ps_parameters, self.cache_enable)
        else:
            success = self.hyper_map_reverse(F.partial(_momentum_opt, self.opt, self.momentum, lr),
                                             gradients, params, moments, self.ps_parameters, self.cache_enable)
        return success


# 实例化网络
net = LeNet5()

# 定义损失函数
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 定义优化器
# net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
net_opt = Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9, weight_decay=0.0005)


# 设置模型保存参数
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 应用模型保存参数
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

def train_net(model, epoch_size, data_path, ckpoint_cb, sink_mode):
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 128)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)

def test_net(model, data_path):
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))

train_epoch = 20
mnist_path = "./datasets/MNIST_Data"
dataset_size = 1
lenet = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(lenet, train_epoch, mnist_path, ckpoint, False)
test_net(lenet, mnist_path)
