import collections

import os
import time
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore import Tensor
from mindspore import jit
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

class SelectiveBackPropagation:
    def __init__(self, compute_losses_func, update_weights_func, optimizer, model,
                 batch_size, epoch_length, loss_selection_threshold=False):
        self.loss_selection_threshold = loss_selection_threshold
        self.compute_losses_func = compute_losses_func
        self.update_weights_func = update_weights_func
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model

        self.loss_hist = collections.deque([], maxlen=batch_size*epoch_length)
        self.selected_inputs, self.selected_targets = [], []
    
    def selective_back_propagation(self, loss_per_img, data, targets):
        effective_batch_loss = None

        cpu_losses = loss_per_img.asnumpy()
        self.loss_hist.extend(cpu_losses.tolist())
        np_cpu_losses = cpu_losses

        selection_probabilities = self._get_selection_probabilities(np_cpu_losses)
        selection = selection_probabilities > np.random.random(*selection_probabilities.shape)

        if self.loss_selection_threshold:
            higher_thres = np_cpu_losses > self.loss_selection_threshold
            selection = np.logical_or(higher_thres, selection)

        selected_losses = []
        for idx in np.argwhere(selection).flatten():
            selected_losses.append(np_cpu_losses[idx])

            # 将 numpy.int64 转换为 Python int,MindSpore 的 __getitem__ 方法不支持 numpy.int64 类型的索引
            idx_int = int(idx)  

            self.selected_inputs.append(data[idx_int].asnumpy())
            self.selected_targets.append(targets[idx_int].asnumpy())

            if len(self.selected_targets) == self.batch_size:
                self.model.set_train()
                input_tensor = Tensor(np.stack(self.selected_inputs), dtype=mstype.float32)
                target_tensor = Tensor(np.stack(self.selected_targets), dtype=mstype.int32)
                
                def forward_fn(data, target):
                    logits = self.model(data)
                    loss = self.compute_losses_func(logits, target)
                    return loss.mean()  
                
                grad_fn = mindspore.value_and_grad(forward_fn, None, self.optimizer.parameters)
                mean_loss, grads = grad_fn(input_tensor, target_tensor)
                self.optimizer(grads)

                effective_batch_loss = mean_loss

                self.model.set_train(False)
                self.selected_inputs = []
                self.selected_targets = []

        return effective_batch_loss

    def _get_selection_probabilities(self, loss):
        percentiles = self._percentiles(self.loss_hist, loss)
        return percentiles ** 2

    def _percentiles(self, hist_values, values_to_search):
        hist_values, values_to_search = np.asarray(hist_values), np.asarray(values_to_search)

        percentiles_values = np.percentile(hist_values, range(100))
        sorted_loss_idx = sorted(range(len(values_to_search)), key=lambda k: values_to_search[k])
        counter = 0
        percentiles_by_loss = [0] * len(values_to_search)
        for idx, percentiles_value in enumerate(percentiles_values):
            while counter < len(values_to_search) and values_to_search[sorted_loss_idx[counter]] < percentiles_value:
                percentiles_by_loss[sorted_loss_idx[counter]] = idx
                counter += 1
                if counter == len(values_to_search): break
            if counter == len(values_to_search): break
        return np.array(percentiles_by_loss)/100

def train(model, data_loader, optimizer, epochs):
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    selective_bp = SelectiveBackPropagation(
        criterion,
        lambda loss: loss.mean(),
        optimizer,
        model,
        batch_size=data_loader.get_batch_size(),
        epoch_length=data_loader.get_dataset_size()
    )
    
    for epoch in range(epochs):
        model.set_train(False)
        
        for batch_idx, (data, target) in enumerate(data_loader.create_tuple_iterator()):
            output = model(data)
            not_reduced_loss = criterion(output, target)
            selective_bp.selective_back_propagation(not_reduced_loss, data, target)
        print(f'Epoch: {epoch+1}/{epochs}')

def standard_train(model, data_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.set_train()
        
        for batch_idx, (data, target) in enumerate(data_loader.create_tuple_iterator()):
            def forward_fn(data, target):
                logits = model(data)
                loss = criterion(logits, target)
                return loss

            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)
            loss, grads = grad_fn(data, target)
            optimizer(grads)
        print(f'Epoch: {epoch+1}/{epochs}')

def evaluate(model, test_loader, device):
    model.set_train(False)
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')
    argmax = ops.Argmax(axis=1)
    
    for data, target in test_loader.create_tuple_iterator():
        output = model(data)
        test_loss += criterion(output, target).asnumpy().item()
        pred = argmax(output)
        correct += (pred == target).asnumpy().sum()
        total += target.shape[0]

    test_loss /= total
    accuracy = 100. * correct / total
    return test_loss, accuracy

def download_cifar10(target_dir="./data_ms"):
    """下载CIFAR-10数据集并解压到目标目录"""
    import os
    import urllib.request
    import tarfile
    
    print("正在下载CIFAR-10数据集...")
    
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # 定义下载URL和目标文件路径
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    filename = os.path.join(target_dir, "cifar-10-binary.tar.gz")
    
    # 下载文件
    urllib.request.urlretrieve(url, filename)
    print(f"下载完成: {filename}")
    
    # 解压文件
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=target_dir)
    print(f"解压完成: {os.path.join(target_dir, 'cifar-10-batches-bin')}")
    
    return True

class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='same')
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
            self.fc1 = nn.Dense(16 * 6 * 6, 120)  # 修改为6*6而不是5*5
            self.fc2 = nn.Dense(120, 84)
            self.fc3 = nn.Dense(84, 10)
            self.relu = nn.ReLU()
            
        def construct(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 16 * 6 * 6)  # 修改为576
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

if __name__ == "__main__":   
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    # 检查并下载CIFAR-10数据集
    #data_dir = "./data_ms"
    #cifar_dir = os.path.join(data_dir, "cifar-10-batches-bin")
    
    #if not os.path.exists(cifar_dir):
    #    download_cifar10(data_dir)
    
    def create_dataset(batch_size=64, training=True):

        cifar_ds_dir = "./data_ms/cifar-10-batches-bin"
        import os
        if not os.path.exists(cifar_ds_dir):
            os.makedirs(cifar_ds_dir)
    
        dataset = ds.Cifar10Dataset(
            dataset_dir=cifar_ds_dir,
            usage='train' if training else 'test',
            shuffle=training,
        )
        
        # 定义数据增强操作
        if training:
            transform_img = [
                vision.RandomCrop((32, 32), (4, 4, 4, 4)),
                vision.RandomHorizontalFlip(),
                vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                vision.HWC2CHW()
            ]
        else:
            transform_img = [
                vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                vision.HWC2CHW()
            ]
            
        transform_label = [transforms.TypeCast(mstype.int32)]  # 确保transform_label不为空，添加剂简单的转换
        dataset = dataset.map(operations=transform_img, input_columns="image")
        dataset = dataset.map(operations=transform_label, input_columns="label")
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset
    
    trainloader = create_dataset(batch_size=64, training=True)
    testloader = create_dataset(batch_size=64, training=False)
    
    epochs = 10
    lr = 0.001
    
    # 使用选择性反向传播训练
    print("\n=== Training with Selective BackPropagation ===")
    model_sb = ConvNet()
    optimizer_sb = nn.Adam(model_sb.trainable_params(), learning_rate=lr)
    
    start_time = time.time()
    train(model_sb, trainloader, optimizer_sb, epochs=epochs)
    sb_train_time = time.time() - start_time
    
    sb_loss, sb_acc = evaluate(model_sb, testloader, None)
    print(f"Selective BP - Training time: {sb_train_time:.2f}s, Test accuracy: {sb_acc:.2f}%, Loss: {sb_loss:.2f}")