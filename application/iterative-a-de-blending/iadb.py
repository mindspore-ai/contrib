import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import FakeImageDataset
from mindspore.nn.optim import Adam
import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
import os
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

def create_dataset(batch_size):
    dataset = FakeImageDataset(num_images=1000, image_size=(64, 64, 3), num_classes=10)
    dataset = dataset.map(input_columns=["image"], operations=[
        vision.Resize((64, 64)),
        vision.RandomHorizontalFlip(0.5),
        vision.AdjustBrightness(0.1),
        vision.HWC2CHW(),
        transforms.TypeCast(ms.float32)
    ])
    dataset = dataset.batch(batch_size)
    return dataset
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")

class SimpleUNet2D(nn.Cell):
    def __init__(self):
        super(SimpleUNet2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, pad_mode="pad", padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, pad_mode="pad", padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, pad_mode="pad", padding=1)
        self.deconv1 = nn.Conv2dTranspose(512, 256, 3, pad_mode="pad", padding=1)
        self.deconv2 = nn.Conv2dTranspose(256, 128, 3, pad_mode="pad", padding=1)
        self.deconv3 = nn.Conv2dTranspose(128, 3, 3, pad_mode="pad", padding=1)
        self.relu = nn.ReLU()

    def construct(self, x, alpha):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        return self.deconv3(x)

def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = t / nb_step
        alpha_end = (t + 1) / nb_step

        d = model(x_alpha, Tensor(alpha_start, ms.float32))
        x_alpha = x_alpha + (alpha_end - alpha_start) * d

    return x_alpha

device = ms.context.get_context("device_target")
model = SimpleUNet2D()
optimizer = Adam(model.trainable_params(), learning_rate=0.001)
loss_fn = nn.MSELoss()

dataloader = create_dataset(batch_size=64)

nb_iter = 0
print('Start training')
for current_epoch in range(100):
    for data in dataloader.create_dict_iterator():
        x1 = (data["image"] * 2) - 1
        x0 = ops.StandardNormal()(x1.shape)

        alpha = ops.UniformReal()((x0.shape[0],))
        x_alpha = alpha.view(-1, 1, 1, 1) * x1 + (1 - alpha).view(-1, 1, 1, 1) * x0

        d = model(x_alpha, alpha)
        loss = loss_fn(d, x1 - x0)

        loss.backward()
        nb_iter += 1

        if nb_iter % 200 == 0:
            print(f'Save export {nb_iter}')
            sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
            save_checkpoint(model, f'celeba_{nb_iter}.ckpt')
            os.makedirs('export', exist_ok=True)
            np.save(f'export/export_{str(nb_iter).zfill(8)}.npy', sample.asnumpy())