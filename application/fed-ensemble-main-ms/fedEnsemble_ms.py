import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import random
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
import itertools
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


# 数据集分割
def split_dataset(dataset, num_parties, batch_size, shuffle=True):
    """
    Split a given dataset into a number of parties, each containing a part of the dataset.
    Create data loaders for each party.
    """
    dataset_size = dataset.get_dataset_size()
    data_indices = list(range(dataset_size))
    split_size = dataset_size // num_parties

    if shuffle:
        random.shuffle(data_indices)

    party_data_indices = [data_indices[i:i + split_size] for i in range(0, dataset_size, split_size)]
    party_dataloaders = []

    for indices in party_data_indices:
        party_dataset = dataset.filter(input_columns=["label"], predicate=lambda x: x in indices)
        party_dataloader = party_dataset.batch(batch_size=batch_size, drop_remainder=True)
        party_dataloaders.append(party_dataloader)

    return party_dataloaders



def client_update(model, optimizer, train_loader, num_epochs):
    """
    Perform a local training update for a client in a federated learning system.
    """
    loss_fn = nn.CrossEntropyLoss()
    def forward_fn(inputs, targets):
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        return loss
    # def train_step(inputs, targets,opt):
    #     loss, grads = grad_fn(inputs, targets)
    #     opt(grads)
    #     return loss

    model.set_train(True)

    # 创建 GradOperation
    # grad_fn = ops.GradOperation(get_by_list=True, sens_param=False)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    params = model.trainable_params()



    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader.create_dict_iterator():
            inputs, labels = data["image"], data["label"]
            labels = ms.Tensor(labels, dtype=ms.int32)

            # 清除梯度
            # optimizer.clear_grad()

            # 前向传播
            # outputs = model(inputs)

            # 计算损失
            # loss = loss_fn(outputs, labels)
            # loss = train_step(outputs, labels,optimizer)
            loss, grads = grad_fn(inputs, labels)
            optimizer(grads)
            # 使用 GradOperation 计算梯度
            # grads = grad_fn(loss, params)

            # 更新参数
            # optimizer(grads)

            running_loss += loss.asnumpy().item()
            print(f"running_loss_total:{running_loss}")
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model



# 测试模型集合
def test_ensemble_model(model_list, test_loader):
    """
    Evaluate an ensemble of deep learning models on a test dataset and calculate the accuracy.
    """
    for model in model_list:
        model.set_train(False)

    correct = 0
    total = 0

    for data in test_loader.create_dict_iterator():
        inputs, labels = data["image"], data["label"]
        predictions = []

        for model in model_list:
            outputs = model(inputs)
            probabilities = ops.softmax(outputs, axis=1)
            predictions.append(probabilities)

        ensemble_predictions = ops.stack(predictions).mean(axis=0)
        _, predicted = ops.argmax(ensemble_predictions, axis=1)

        total += labels.size
        correct += (predicted == labels).sum().asnumpy().item()

    accuracy = 100 * correct / total
    print(f"Ensemble Accuracy: {accuracy:.2f}%")
    return accuracy


# 模型聚合
def aggregate_models(models, global_model, num_clients):
    """
    Aggregate models from multiple clients using Federated Averaging (FedAvg) algorithm.
    """
    global_dict = global_model.parameters_dict()

    for key in global_dict.keys():
        update_dict = ops.stack(
            [model.parameters_dict()[key].astype(nn.float32) - global_dict[key].astype(nn.float32) for model in models],
            0).sum(axis=0)
        global_dict[key] = global_dict[key] + update_dict * (1 / num_clients)

    global_model.set_parameters(global_dict)
    return global_model


# 创建排列矩阵
def create_permutation_matrix(q, k):
    """
    Create a permutation matrix of size q x k.
    """
    perm_matrix = np.zeros((q, k))

    for i in range(q):
        perm_matrix[i] = np.random.permutation(k)

    perm_matrix = perm_matrix.astype(int).tolist()
    return perm_matrix


# 实现 Fed-Ensemble 算法
def fed_ensemble(C, T, A, P, test_dataloader, learning_rate, num_epochs, Q):
    """
    Implement the Fed-Ensemble algorithm for federated learning.
    """

    # opt = [nn.SGD(model.trainable_params(), learning_rate=learning_rate) for model in A]
    opt=[]
    step_size=390
    learning_rate = nn.cosine_decay_lr(min_lr=0.00001,
                                       max_lr=0.001,
                                       total_step=step_size * 10,
                                       step_per_epoch=step_size,
                                       decay_epoch=10)
    for model in A:
        # 获取模型的可训练参数
        params = model.trainable_params()

        # 创建优化器
        optimizer = nn.SGD(params, learning_rate=learning_rate)
        # 将优化器添加到列表中
        opt.append(optimizer)

    clients_per_stratum = len(C) // Q
    strata = [C[i:i + clients_per_stratum] for i in range(0, len(C), clients_per_stratum)]
    permutation_matrix = create_permutation_matrix(Q, len(A))

    for t in range(T):
        for r in range(len(A)):
            client_models = [[] for _ in range(len(A))]
            for q in range(Q):
                selected_clients = random.sample(strata[q], P)

                model = A[permutation_matrix[q][r]]
                for client in selected_clients:
                    client_models[permutation_matrix[q][r]].append(
                        client_update(model, opt[permutation_matrix[q][r]], client, num_epochs))

            models = [aggregate_models(client_models[agg], A[agg], len(C)) for agg in range(len(A))]

        test_ensemble_model(models, test_dataloader)

    return models


# 主函数
def train(num_participants, num_models_in_ensemble, num_selected_participants, num_training_ages, Model,
          training_dataset,
          test_dataset, samples_per_batch, num_epoch, l_r, num_strata):
    train_dataloaders_clients = []
    for i in range(num_participants):
        train_dataloaders_clients.append(
            ds.GeneratorDataset(training_dataset, ["image", "label"]).batch(samples_per_batch, drop_remainder=True))

    test_dataloader = ds.GeneratorDataset(test_dataset, ["image", "label"]).batch(samples_per_batch,
                                                                                  drop_remainder=True)

    models_in_ensemble = [Model() for _ in range(num_models_in_ensemble)]
    ensemble = fed_ensemble(train_dataloaders_clients, num_training_ages, models_in_ensemble, num_selected_participants,
                            test_dataloader, l_r, num_epoch, num_strata)

    for idx, aggregator in enumerate(ensemble):
        ms.save_checkpoint(aggregator, f'ensemble_model_{idx}.ckpt')