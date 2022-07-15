"""
Filename: communication_gan.py
Author: fangxiuwen
Contact: fangxiuwen67@163.com
"""
from option import args_parser
from models import Cnn2layerfcModel
from model_utils import get_model_list, EarlyStop, NLLLoss, mkdirs
from data_utils import Femnist, FemnistValTest, pre_handle_femnist_mat, generate_partial_femnist, \
    generate_bal_private_data, get_mnist_dataset, get_device_id, get_device_num
import mindspore
import mindspore.nn as nn
from mindspore.dataset import transforms
from mindspore.dataset import vision
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn import Accuracy
from mindspore import Model
import mindspore.context as context
import mindspore.dataset as ds
from mindspore.communication.management import init



def train_models_mnist(models_mnist_list, lr, optimizer, epochs):
    '''
    Train an array of models on the mnist dataset.
    '''
    for n, model in enumerate(models_mnist_list):
        train, val, _ = get_mnist_dataset()
        train = train.batch(batch_size=256, drop_remainder=True)
        val = val.batch(batch_size=256, drop_remainder=True)
        train_models_mnist_bug(n, model, train, val, lr, optimizer, epochs)


def train_models_mnist_bug(n, model, train, val, lr, optimizer, epochs):
    '''
    Train an array of models on the mnist dataset.
    '''
    print("Training model", n)
    dataset_sink = context.get_context('device_target') == 'CPU'
    steps_per_epoch = train.get_dataset_size()

    # 定义优化器
    if optimizer == 'sgd':
        optimizer = nn.SGD(model.trainable_params(), learning_rate=lr, momentum=0.5)
    elif optimizer == 'adam':
        optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=1e-4)

    # 定义损失函数
    criterion = NLLLoss()
    ckpt_cfg = CheckpointConfig(saved_network=model)
    ckpt_cb = ModelCheckpoint(prefix="mnist_model", directory='Network/Mnist_model', config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)
    mnist_model = Model(model, criterion, optimizer, metrics={"Accuracy": Accuracy()})

    mnist_model.train(epochs, train, callbacks=[ckpt_cb, loss_cb], dataset_sink_mode=dataset_sink)
    mindspore.save_checkpoint(model, "./Model/Mnist_model_"+str(n)+".ckpt")
    acc = mnist_model.eval(val, dataset_sink_mode=dataset_sink)

    print("{}".format(acc))
    print('End Training model', n, 'on mnist')


def train_models_bal_femnist(models_femnist_list, train, val_x, val_y, lr, optimizer, epochs):
    '''
    Train an array of models on the femnist dataset.
    '''
    for n, model in enumerate(models_femnist_list):
        train_models_bal_femnist_bug(n, model, optimizer, lr, train, val_x, val_y, epochs)


def train_models_bal_femnist_bug(n, model, optimizer, lr, train, val_x, val_y, epochs):
    '''
    Train an array of models on the femnist dataset.
    '''
    print('train Local Model {} on Private Dataset'.format(n))
    dataset_sink = context.get_context('device_target') == 'CPU'
    # Define optimizer
    if optimizer == 'sgd':
        optimizer = nn.SGD(model.trainable_params(), learning_rate=lr, momentum=0.5)
    elif optimizer == 'adam':
        optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=1e-4)

    # Define loss function
    criterion = NLLLoss()

    apply_transform = transforms.py_transforms.Compose([vision.py_transforms.ToTensor(),
                                                        vision.py_transforms.Normalize((0.1307,), (0.3081,))])

    femnist_bal_data_train = Femnist(train[n], apply_transform)
    femnist_bal_data_validation = FemnistValTest(val_x, val_y, apply_transform)

    trainloader = ds.GeneratorDataset(femnist_bal_data_train, ["data", "label"], shuffle=True)
    valloader = ds.GeneratorDataset(femnist_bal_data_validation, ["data", "label"], shuffle=True)
    trainloader = trainloader.batch(batch_size=32, drop_remainder=True)
    valloader = valloader.batch(batch_size=128, drop_remainder=True)

    print('Begin Training model', n, 'on Femnist')
    steps_per_epoch = trainloader.get_dataset_size()

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=10)
    ckpt_cb = ModelCheckpoint(prefix="femnist_model", directory='Network/Femnist_model', config=ckpt_cfg)
    loss_cb = LossMonitor(steps_per_epoch)
    stop_cb = EarlyStop(control_loss=0.5)

    femnist_model = Model(model, criterion, optimizer, metrics={"Accuracy": Accuracy()})
    femnist_model.train(epochs, trainloader, callbacks=[ckpt_cb, loss_cb, stop_cb], dataset_sink_mode=dataset_sink)
    mindspore.save_checkpoint(model, "./Model/Femnist_model_"+str(n)+".ckpt")
    acc = femnist_model.eval(valloader, dataset_sink_mode=dataset_sink)

    print("{}".format(acc))
    print('End Training model', n, 'on femnist')


def test_models_public(models_public_list):
    """
    Test models on public dataset
    """
    dataset_sink = context.get_context('device_target') == 'CPU'
    accuracy_list = []
    loss = NLLLoss()
    for _, model in enumerate(models_public_list):
        _, _, test = get_mnist_dataset()
        test = test.batch(batch_size=256, drop_remainder=True)
        model = Model(model, loss_fn=loss, metrics={"Accuracy": Accuracy()})
        acc = model.eval(test, dataset_sink_mode=dataset_sink)
        accuracy_list.append(acc)
    print(accuracy_list)


def test_models_femnist(models_femnist_list, test_x, test_y):
    """
    Test models on private dataset
    """
    dataset_sink = context.get_context('device_target') == 'CPU'
    apply_transform = transforms.py_transforms.Compose([vision.py_transforms.ToTensor(),
                                                        vision.py_transforms.Normalize((0.1307,), (0.3081,))])
    femnist_bal_data_test = FemnistValTest(test_x, test_y, apply_transform)
    testloader = ds.GeneratorDataset(femnist_bal_data_test, ["data", "label"], shuffle=True)
    testloader = testloader.batch(batch_size=128)

    accuracy_list = []
    loss = NLLLoss()
    for _, model in enumerate(models_femnist_list):
        model = Model(model, loss, metrics={"accuracy"})
        acc = model.eval(testloader, dataset_sink_mode=dataset_sink)
        accuracy_list.append(acc)
    print(accuracy_list)


if __name__ == '__main__':
    args = args_parser()
    # device ='cuda' if args.gpu else 'cpu'
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target=args.device_target)
    device = 'Ascend'
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target)
    device_num = get_device_num()
    if args.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel',
                                              gradients_mean=True)
            init()
    models = {"2_layer_CNN": Cnn2layerfcModel}  # 字典的函数类型
    models_ini_list = [{"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 256}},
                       {"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 384}},
                       {"model_type": "2_layer_CNN", "params": {"n1": 128, 'n2': 512}},
                       {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 256}},
                       {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 512}}
                       ]
    models_list = []
    mkdirs('./Model')

    # Initialize the model sequence
    for epoch in range(5):
        tempmodel = models[models_ini_list[epoch]["model_type"]](models_ini_list[epoch]["params"])
        models_list.append(tempmodel)

    # Train to convergence on mnist
    class TrainMnistparams:
        lr = 0.001
        optimizer = 'adam'
        epochs = 10

    train_models_mnist(models_mnist_list=models_list, lr=TrainMnistparams.lr, optimizer=TrainMnistparams.optimizer,
                       epochs=TrainMnistparams.epochs)

    root = "./Model/"
    name = ["Mnist_model_0.ckpt", "Mnist_model_1.ckpt", "Mnist_model_2.ckpt",
            "Mnist_model_3.ckpt", "Mnist_model_4.ckpt"]
    # Transfer model parameters into the param dictionary
    mnist_models_list = get_model_list(root=root, name=name, models_ini_list=models_ini_list, models=models)
    test_models_public(models_public_list=mnist_models_list)

    # Train to convergence on femnist
    root = "./Model/"
    name = ["Mnist_model_0.ckpt", "Mnist_model_1.ckpt", "Mnist_model_2.ckpt",
            "Mnist_model_3.ckpt", "Mnist_model_4.ckpt"]
    mnist_models_list = get_model_list(root=root, name=name, models_ini_list=models_ini_list, models=models)
    x_train, y_train, _, x_test, y_test, _ = pre_handle_femnist_mat()
    y_train += len(args.public_classes)
    y_test += len(args.public_classes)

    femnist_x_test, femnist_y_test = generate_partial_femnist(x=x_test, y=y_test, class_in_use=args.private_classes,
                                                              verbose=False)
    private_bal_femnist_data = generate_bal_private_data(x=x_train, y=y_train, n_parties=args.N_parties,
                                                         classes_in_use=args.private_classes,
                                                         n_samples_per_class=args.N_samples_per_class,
                                                         data_overlap=False)
    test_models_femnist(models_femnist_list=mnist_models_list, test_x=femnist_x_test, test_y=femnist_y_test)

    class TrainFemnistParams:
        lr = 0.001
        optimizer = 'adam'
        epochs = 10

    train_models_bal_femnist(models_femnist_list=mnist_models_list, train=private_bal_femnist_data,
                             val_x=femnist_x_test, val_y=femnist_y_test, lr=TrainFemnistParams.lr,
                             optimizer=TrainFemnistParams.optimizer, epochs=TrainFemnistParams.epochs)

    root = "./Model/"
    name = ["Femnist_model_0.ckpt", "Femnist_model_1.ckpt", "Femnist_model_2.ckpt",
            "Femnist_model_3.ckpt", "Femnist_model_4.ckpt"]
    femnist_models_list = get_model_list(root=root, name=name, models_ini_list=models_ini_list, models=models)
    print("Test on Femnist:")
    test_models_femnist(models_femnist_list=femnist_models_list, test_x=femnist_x_test, test_y=femnist_y_test)
    print("Test on Mnist:")
    test_models_public(models_public_list=femnist_models_list)
