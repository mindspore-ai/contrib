"""
Filename: communication_gan.py
Author: fangxiuwen
Contact: fangxiuwen67@163.com
"""
import os
from data_utils import Femnist, FemnistValTest, Mydata, pre_handle_femnist_mat, generate_bal_private_data
from option import args_parser
from model_utils import NLLLoss, average_weights, mkdirs
from models import DomainIdentifier
import numpy as np
import mindspore
from mindspore.dataset import transforms, vision
import mindspore.dataset as ds
from mindspore import nn, Tensor, DatasetHelper, save_checkpoint, ops, Parameter, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import copy
import pdb
from mindspore import dtype as mstype
from tqdm import tqdm

def MNIST_random(dataset, epochs, num_item=5000):
    """
    Divide MNIST
    """
    dict_epoch, all_idxs = {}, [i for i in range(48000)]
    for i in range(epochs):
        dict_epoch[i] = set(np.random.choice(all_idxs, num_item, replace=False))
        all_idxs = list(set(all_idxs) - dict_epoch[i])
    return dict_epoch


def list_add(a, b):
    """
    Add operation
    """
    c = []
    for i in range(len(a)):
        d = []
        for j in range(len(a[i])):
            d.append(a[i][j] + b[i][j])
        c.append(d)
    return c


def get_avg_result(temp_sum_result, num_client):
    """
    Calculate average result
    """
    for itemx in range(len(temp_sum_result)):
        for itemy in range(len(temp_sum_result[itemx])):
            temp_sum_result[itemx][itemy] /= num_client
    return temp_sum_result


class DatasetSplit:
    """
    An abstract Dataset class wrapped around Mindspore Dataset class
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        index = self.idxs[item]
        image, label = self.dataset[index]
        image = Tensor(image)
        label = Tensor(label)
        return image, label


class DomainDataset:
    """
    An abstract Dataset class wrapped around Mindspore Dataset class
    """
    def __init__(self,publicadataset,privatedataset,localindex,step1=True):
        imgs = []
        if step1:
            for index in range(len(publicadataset)):
                imgs.append((publicadataset[index][0],10))
            for index in range(len(privatedataset)):
                imgs.append((privatedataset[index][0],localindex))
        else:
            for index in range(len(publicadataset)):
                imgs.append((publicadataset[index][0],localindex))
            for index in range(len(privatedataset)):
                imgs.append((privatedataset[index][0],10))
        self.imgs = imgs
    def __getitem__(self, index):
        image, domain_label = self.imgs[index]
        return image, domain_label
    def __len__(self):
        return len(self.imgs)


def train_models_collaborate_gan(device, models_list, train, user_number, collaborative_epoch,
                                 output_classes):
    mkdirs('./Model/final_model')
    epoch_groups = MNIST_random(dataset=train, epochs=collaborative_epoch)

    train_loss = []
    test_accuracy = []
    for i in range(user_number):
        train_loss.append([])
    for i in range(user_number):
        test_accuracy.append([])

    for epoch in range(collaborative_epoch):
        train_batch_loss = []
        for i in range(user_number):
            train_batch_loss.append([])
        """
        Create dataset
        """
        train_dataset = [data for data in train]
        train_split = DatasetSplit(train_dataset, list(epoch_groups[epoch]))
        trainloader = ds.GeneratorDataset(train_split, ["data", "label"], shuffle=True)
        trainloader = trainloader.batch(batch_size=256)
        trainloader = DatasetHelper(trainloader, dataset_sink_mode=False)


        """
        Start training
        """
        for batch_idx, (image, label) in enumerate(tqdm(trainloader)):
            label = Tensor(label, dtype=mindspore.int32)
            image = Tensor(image, dtype=mindspore.float32)
            temp_sum_result = [ [] for _ in range(len(label))]
            for item in range(len(temp_sum_result)):
                for i in range(output_classes):
                    temp_sum_result[item].append(0)

            # Make output together
            for n, model in enumerate(models_list):
                    model.set_train(mode=False)
                    outputs = model(image)
                    pred_labels = outputs.asnumpy().tolist() # 转成list
                    temp_sum_result = list_add(pred_labels,temp_sum_result) # 把每次的结果都给加到一起

            temp_sum_result = get_avg_result(temp_sum_result,user_number) # 根据参与训练的时候用户把结果除以对应的数量
            labels = Tensor(temp_sum_result, dtype=mindspore.int32)
            lr = 0.001
            optimizer = 'adam'
            for n,model in enumerate (models_list):
                train_models_collaborate_bug_gan(model,device,optimizer,lr,image,labels,batch_idx,n,epoch,trainloader)


def train_models_collaborate_bug_gan(model, device, optimizer, lr, images, labels, batch_idx, n, epoch,
                                     trainloader):
    modelurl = './Model/final_model'

    model.set_train(mode=True)
    """
    Define loss function and optimizer
    """
    criterion = nn.L1Loss(reduction='mean')
    if optimizer == 'sgd':
        optimizer = nn.SGD(params=model.trainable_params(), learning_rate=lr, momentum=0.5)
    elif optimizer == 'adam':
        optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr, weight_decay=1e-4)

    """
    Start training
    """
    outputs = model(images)
    loss = criterion(outputs, labels)
    weights = mindspore.ParameterTuple(optimizer.parameters)
    grad = ops.GradOperation(get_by_list=True)
    grads = grad(model, weights)(images)
    loss = ops.Depend()(loss, optimizer(grads))
    if batch_idx % 10 == 0:
        print('Collaborative traing : Local Model {} Train Epoch: {} Loss: {}'.format(n, epoch + 1, loss))
    """
    Save model
    """
    save_checkpoint(model, modelurl + '/LocalModel{}.ckpt'.format(n))


def train_models_bal_femnist_collaborate(device, models_list, modelurl):
    class train_params:
        lr = 0.001
        optimizer = 'adam'
        epochs = 1
    args = args_parser()

    """
    Create dataset
    """
    X_train, y_train, writer_ids_train, X_test, y_test, writer_ids_train, writer_ids_test = pre_handle_femnist_mat()
    y_train += len(args.public_classes)
    y_test += len(args.public_classes)
    private_bal_femnist_data, total_private_bal_femnist_data = \
        generate_bal_private_data(X=X_train, y=y_train, N_parties=args.N_parties,
                                  classes_in_use=args.private_classes,
                                  N_samples_per_class=args.N_samples_per_class, data_overlap=False)

    for n, model in enumerate(models_list):
        train_models_bal_femnist_bug(device, n, model, train_params.optimizer, train_params.lr,
                                     private_bal_femnist_data, train_params.epochs, modelurl)


def train_models_bal_femnist_bug(device, n, model, optimizer, lr, train, epochs, modelurl):
    print('train Local Model {} on Private Dataset'.format(n))

    """
    Define loss function and optimizer
    """
    criterion = NLLLoss()
    if optimizer == 'sgd':
        optimizer = nn.SGD(params=model.trainable_params(), learning_rate=lr, momentum=0.5)
    elif optimizer == 'adam':
        optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr, weight_decay=1e-4)

    """
    Generate trainloader
    """
    apply_transform = transforms.py_transforms.Compose([vision.py_transforms.ToTensor(),
                                                        vision.py_transforms.Normalize((0.1307,), (0.3081,))])
    femnist_bal_data_train = Mydata(train[n], apply_transform)
    trainloader = ds.GeneratorDataset(femnist_bal_data_train, ["data", "label"], shuffle=True)
    trainloader = trainloader.batch(batch_size=5)
    trainloader = DatasetHelper(trainloader, dataset_sink_mode=False)

    """
    Start training
    """
    train_epoch_losses = []
    print('Begin Training on Femnist')
    for epoch in range(epochs):
        model.set_train(mode=True)
        train_batch_losses = []
        for batch_idx, (images, labels) in enumerate(tqdm(trainloader)):
            labels = Tensor(labels, dtype=mindspore.int32)
            images = Tensor(images, dtype=mindspore.float32)
            outputs = model(images)
            loss = criterion(outputs, labels)
            weights = mindspore.ParameterTuple(optimizer.parameters)
            grad = ops.GradOperation(get_by_list=True)
            grads = grad(model, weights)(images)
            loss = ops.Depend()(loss, optimizer(grads))
            if batch_idx % 5 == 0:
                print('Local Model {} Train Epoch: {} Loss: {}'.format(n, epoch + 1, loss))
            train_batch_losses.append(loss)
        loss_avg = sum(train_batch_losses) / len(train_batch_losses)
        train_epoch_losses.append(loss_avg)
        """
        Save model
        """
        save_checkpoint(model, modelurl + '/LocalModel{}.ckpt'.format(n))


def feature_domain_alignment(device, train, models_list, modelurl, domain_identifier_epochs,
                             gan_local_epochs):
    """
    Generate the sample indices for each round
    """
    url = 'mnist'
    epoch_groups = MNIST_random(dataset=train, epochs=domain_identifier_epochs, num_item=40)

    args = args_parser()

    """
    Generate femnist
    """
    X_train, y_train, writer_ids_train, X_test, y_test, writer_ids_train, writer_ids_test = pre_handle_femnist_mat()
    y_train += len(args.public_classes)
    y_test += len(args.public_classes)
    private_bal_femnist_data, total_private_bal_femnist_data = \
        generate_bal_private_data(X=X_train, y=y_train, N_parties=args.N_parties,
                                  classes_in_use=args.private_classes,
                                  N_samples_per_class=args.N_samples_per_class, data_overlap=False)
    apply_transform = transforms.py_transforms.Compose([vision.py_transforms.ToTensor(),
                                                        vision.py_transforms.Normalize((0.1307,), (0.3081,))])

    """
    GanStep1
    """
    epoch_loss = []
    for epoch in range(domain_identifier_epochs):
        local_weights,local_losses = [],[]
        for n, model in enumerate(models_list):
            """
            Load GanModel0
            """
            GANmodel = DomainIdentifier()
            param_dict = load_checkpoint("./GanModel0.ckpt")
            load_param_into_net(GANmodel, param_dict)
            GANmodel.set_train(mode=True)

            """
            Create dataset
            """
            femnist_bal_data_train = Mydata(private_bal_femnist_data[n], apply_transform)
            trainlist = [data for data in train]
            public_dataset = DatasetSplit(trainlist, list(epoch_groups[epoch]))
            pubilcdataset_list = [data for data in public_dataset]
            privatedataset_list = [data for data in femnist_bal_data_train]
            traindataset = DomainDataset(publicadataset=pubilcdataset_list, privatedataset=privatedataset_list,
                                         localindex=n, step1=True)
            trainloader = ds.GeneratorDataset(traindataset, ["data", "label"], shuffle=True)
            trainloader = trainloader.batch(batch_size=30)
            trainloader = DatasetHelper(trainloader, dataset_sink_mode=False)

            """
            Define loss function and optimizer
            """
            criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            optimizer = nn.Adam(params=GANmodel.trainable_params(), learning_rate=0.001, weight_decay=1e-4)

            """
            Start training
            """
            batch_loss = []
            for batch_idx,(images,domain_labels) in enumerate(tqdm(trainloader)):
                domain_labels = Tensor(domain_labels, dtype=mindspore.int32)
                images = Tensor(images, dtype=mindspore.float32)
                temp_outputs = model(images,True)
                domain_outputs = GANmodel(temp_outputs,n)
                loss = criterion(domain_outputs,domain_labels)
                weights = mindspore.ParameterTuple(optimizer.parameters)
                grad = ops.GradOperation(get_by_list=True)
                grads = grad(GANmodel, weights)(temp_outputs,n)
                loss = ops.Depend()(loss, optimizer(grads))
                print('Gan Step1 on Model {} Train Epoch: {} Loss: {}'.format(n, epoch + 1, loss))
                batch_loss.append(loss)
            w = GANmodel.parameters_dict()
            local_weights.append(copy.deepcopy(w))
            local_losses.append(sum(batch_loss)/len(batch_loss))

        """
        Calculate and save the average model weight
        """
        epoch_loss.append(sum(local_losses)/len(local_losses))
        global_weights = average_weights(local_weights)
        global_weights_param = {}
        for i in global_weights.keys():
            global_weights_param[i] = Parameter(global_weights[i])
        GANmodel = DomainIdentifier()
        load_param_into_net(GANmodel, global_weights_param)
        save_checkpoint(GANmodel, "./GanModel0.ckpt")

    dirpath = 'Figures/'+url+'/collaborate_gan'
    mkdirs(dirpath)
    file = open(dirpath+'/GanStep1.txt', 'a+')
    file.write(str(epoch_loss)[1:-1])
    file.write('\n')
    file.close()

    """
    GanStep2
    """
    epoch_loss = []
    mkdirs(modelurl + '/collaborate_gan')
    for epoch in range(gan_local_epochs):
        local_losses = []
        for n, model in enumerate(models_list):
            """
            Load GanModel0
            """
            GANmodel = DomainIdentifier()
            param_dict = load_checkpoint("./GanModel0.ckpt")
            load_param_into_net(GANmodel, param_dict)

            """
            Create dataset
            """
            femnist_bal_data_train = Mydata(private_bal_femnist_data[n], apply_transform)
            trainlist = [data for data in train]
            public_dataset = DatasetSplit(trainlist, list(epoch_groups[epoch]))
            pubilcdataset_list = [data for data in public_dataset]
            privatedataset_list = [data for data in femnist_bal_data_train]
            traindataset = DomainDataset(publicadataset=pubilcdataset_list, privatedataset=privatedataset_list,
                                         localindex=n, step1=False)
            trainloader = ds.GeneratorDataset(traindataset, ["data", "label"], shuffle=True)
            trainloader = trainloader.batch(batch_size=30)
            trainloader = DatasetHelper(trainloader, dataset_sink_mode=False)

            """
            Define loss function and optimizer
            """
            criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.0001, weight_decay=1e-4)

            """
            Start training
            """
            batch_loss = []
            model.set_train(mode=True)
            for batch_idx, (images, domain_labels) in enumerate(tqdm(trainloader)):
                domain_labels = Tensor(domain_labels, dtype=mindspore.int32)
                images = Tensor(images, dtype=mindspore.float32)
                temp_outputs = model(images,True)
                domain_outputs = GANmodel(temp_outputs,n)
                loss = criterion(domain_outputs,domain_labels)
                weights = mindspore.ParameterTuple(optimizer.parameters)
                grad = ops.GradOperation(get_by_list=True)
                grads = grad(model, weights)(images,True)
                loss = ops.Depend()(loss, optimizer(grads))
                print('Gan Step2 on Model {} Train Epoch: {} Loss: {}'.format(n, epoch + 1, loss))
                batch_loss.append(loss)

            """
            Save model
            """
            # local_param_list = get_param_list(model)
            save_checkpoint(model, modelurl + '/collaborate_gan/LocalModel{}.ckpt'.format(n))
            local_losses.append(sum(batch_loss)/len(batch_loss))
        epoch_loss.append(sum(local_losses)/len(local_losses))

    dirpath = 'Figures/'+url+'/collaborate_gan'
    mkdirs(dirpath)
    file = open(dirpath+'/GanStep2.txt', 'a+')
    file.write(str(epoch_loss)[1:-1])
    file.write('\n')
    file.close()
