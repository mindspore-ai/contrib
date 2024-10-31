import os
import numpy as np
from PIL import Image


import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import pandas as pd


import warnings
warnings.filterwarnings("ignore")


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1))


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


class Dataset_LMNET(ms.dataset.Dataset):

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            # print(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            if use_transform is not None:
                image = use_transform(image)
            X.append(image.squeeze_(0))
        X = ops.stack(X, dim=0)

        return X  #[28, 3, 122, 122]

    def read_csv(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            lip_data = pd.read_csv(open(os.path.join(path, selected_folder, 'frame_{:03d}.csv'.format(i)), 'r'),header=None)
            lip_data = lip_data.as_matrix()  # juzhen
            lip_data = ms.Tensor(lip_data)  # from juzhen to tensor double
            lip_data = lip_data.float()  # from tensor double to tensor float
            lip_data = lip_data.permute(1, 0)

            #if use_transform is not None:
            #   lip_data = use_transform(lip_data)

            X.append(lip_data)
        X = ops.stack(X, axis=0).permute(1,0,2) # [3, 28, 200])

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        # X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        X = self.read_csv(self.data_path, folder, self.transform)
        y = ms.Tensor([self.labels[index]],dtype=ms.float32)  # (labels) LongTensor are for int64 instead of FloatTensor
        return X, y



from mindspore import load_checkpoint, load_param_into_net

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Cell):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = ops.Flatten()(x)
        x = self.fc(x)

        return x

def resnet34(pretrained=False, num_classes=1000):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        param_dict = load_checkpoint("path_to_pretrained_resnet34.ckpt")
        load_param_into_net(model, param_dict)
    return model


class LMNET(nn.Cell):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, num_classes=68):
        super(LMNET, self).__init__()
        print('------ LMNET model-----')

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        
        ## point
        self.conv1_1 = nn.SequentialCell(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        ## frame
        self.conv1_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
        )

        resnet = resnet34(pretrained=True)
        modules = list(resnet.children())[1:-1]  # delete the last fc layer.
        # print(len(modules))  
        # print(modules)

        self.resnet = nn.SequentialCell(*modules)
        self.fc1 = nn.Dense(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Dense(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Dense(fc_hidden2, num_classes)

        # initial weight
        initial_model_weight(layers=list(self.children()))
        print('weight initial finished!')

    def construct(self, x_3d):
        # ResNet CNN
        x_1 = self.conv1_1(self.W * x_3d + x_3d)
        x_2 = self.conv1_2(x_3d)
        x = self.resnet(ops.cat((x_1,x_2),axis=1))
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = ops.relu(x)
        x = self.bn2(self.fc2(x))
        x = ops.relu(x)
        x = ops.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

import unittest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TestLabelEncoding(unittest.TestCase):

    def setUp(self):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.labels = ['class1', 'class2', 'class1', 'class3']
        self.label_encoder.fit(self.labels)
        self.onehot_encoder.fit(self.label_encoder.transform(self.labels).reshape(-1, 1))

    def test_labels2cat(self):
        cat_labels = labels2cat(self.label_encoder, self.labels)
        expected_cat_labels = self.label_encoder.transform(self.labels)
        np.testing.assert_array_equal(cat_labels, expected_cat_labels, "labels2cat function failed")

    def test_labels2onehot(self):
        onehot_labels = labels2onehot(self.onehot_encoder, self.label_encoder, self.labels)
        expected_onehot_labels = self.onehot_encoder.transform(self.label_encoder.transform(self.labels).reshape(-1, 1))
        np.testing.assert_array_equal(onehot_labels, expected_onehot_labels, "labels2onehot function failed")

    def test_onehot2labels(self):
        labels = onehot2labels(self.label_encoder, self.onehot_encoder.transform(self.label_encoder.transform(self.labels).reshape(-1, 1)))
        expected_labels = self.labels
        self.assertEqual(labels, expected_labels, "onehot2labels function failed")

    def test_cat2labels(self):
        labels = cat2labels(self.label_encoder, self.label_encoder.transform(self.labels))
        expected_labels = self.labels
        self.assertEqual(labels, expected_labels, "cat2labels function failed")

if __name__ == '__main__':
    unittest.main()
