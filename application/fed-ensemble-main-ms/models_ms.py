import mindspore.nn as nn
import mindspore as ms
from resnet50 import resnet50
import mindspore.ops as ops
class BaseModel(nn.Cell):
    def __init__(self, hidden_size = 512, num_classes = 10):
        super(BaseModel, self).__init__()

        self.model = resnet50(pretrained=False)
        self.fc1 = nn.Dense(2048, hidden_size)
        self.fc2 = nn.Dense(hidden_size, num_classes)

        self.relu=nn.ReLU()
        self.flatten=ops.Flatten()

    def construct(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        x = self.model.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
