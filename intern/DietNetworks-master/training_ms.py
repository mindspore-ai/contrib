import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from tqdm import tqdm as tqdm_func
import pandas as pd
import numpy as np
from mindspore.nn.optim import Adam

# 设置全局上下文
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU") 

criterion1 = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
criterion2 = nn.MSELoss()

ALPHA = 1   
BETA = 0.000000001  

def train_pass_diet(model, data_loader, optimizer, device=None):
    model.set_train()

    pass_loss = 0.0
    total = 0.0
    correct = 0.0

    for inputs, labels in data_loader:
        inputs = Tensor(inputs.asnumpy(), dtype=ms.float32)
        labels = Tensor(labels.asnumpy(), dtype=ms.int32)

        def forward_fn(inputs, labels):
            x_hat, y_hat = model(inputs)
            loss1 = criterion1(y_hat, labels)
            loss2 = criterion2(x_hat, inputs)
            loss = ALPHA * loss1 + BETA * loss2
            return loss, x_hat, y_hat

        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        (loss, x_hat, y_hat), grads = grad_fn(inputs, labels)
        
        optimizer(grads)

        pass_loss += float(loss.asnumpy())
        
        argmax = ops.Argmax(axis=1)
        predicted = argmax(y_hat)
        predicted_np = predicted.asnumpy()
        labels_np = labels.asnumpy()
        
        total += len(labels_np)
        correct += (predicted_np == labels_np).sum()
        
    accuracy = 100 * correct / total

    return pass_loss / len(data_loader), accuracy


def test_pass_diet(model, data_loader, device=None):
    model.set_train(False)
    
    pass_loss = 0.0
    total = 0.0
    correct = 0.0
    
    for inputs, labels in data_loader:
        inputs = Tensor(inputs.asnumpy(), dtype=ms.float32)
        labels = Tensor(labels.asnumpy(), dtype=ms.int32)
        
        x_hat, y_hat = model(inputs)

        loss1 = criterion1(y_hat, labels)
        loss2 = criterion2(x_hat, inputs)
        loss = ALPHA * loss1 + BETA * loss2
        
        pass_loss += float(loss.asnumpy())
        
        argmax = ops.Argmax(axis=1)
        predicted = argmax(y_hat)
        
        # 转换为NumPy数组以进行评估
        predicted_np = predicted.asnumpy()
        labels_np = labels.asnumpy()
        
        total += len(labels_np)
        correct += (predicted_np == labels_np).sum()           
        accuracy = 100 * correct / total
    
        return pass_loss / len(data_loader), accuracy


def run_training_diet(model, nb_epochs, train_loader, test_loader, device=None, lr=0.001):  
    optimizer = Adam(model.trainable_params(), learning_rate=lr)
    progress_bar = tqdm_func(range(nb_epochs))
    loss_history = []
    best_acc = 0

    for epoch in progress_bar:
        train_loss, train_acc = train_pass_diet(model, train_loader, optimizer)
        test_loss, test_acc = test_pass_diet(model, test_loader)
        
        loss_history.append(
            {"loss": train_loss, "accuracy": train_acc, "set": "train", "epochs": epoch}
        )
        loss_history.append(
            {"loss": test_loss, "accuracy": test_acc, "set": "test", "epochs": epoch}
        )
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    print("best test accuracy:", best_acc)

    return pd.DataFrame(loss_history)