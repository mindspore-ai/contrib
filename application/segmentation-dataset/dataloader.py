import os
import PIL.Image as Image
import numpy as np

import mindspore
from mindspore.dataset import GeneratorDataset


class BCSDDataset(object):
    def __init__(self, path, transforms=None,  ext='.jpeg'):
        self.transforms = transforms

        path_x = os.path.join(path, 'X')
        self.images = sorted(os.listdir(path_x))
        self.images = filter(lambda x: x.startswith(
            'X') and x.endswith(ext), self.images)
        self.images = [os.path.join(path_x, im) for im in self.images]

        path_y = os.path.join(path, 'y')
        self.labels = sorted(os.listdir(path_y))
        self.labels = filter(lambda x: x.startswith(
            'y') and x.endswith(ext), self.labels)
        self.labels = [os.path.join(path_y, im) for im in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.images[idx]).convert(
            'RGB').resize((512, 512)))
        y = np.array(Image.open(self.labels[idx]).convert(
            'L').resize((512, 512)))
        y[y != 0] = 1

        if self.transforms is not None:
            x = self.transforms(x)
            y = self.transforms(y)

        return x, y


if __name__ == "__main__":
    # https://www.kaggle.com/datasets/saifkhichi96/bank-checks-signatures-segmentation-dataset
    BCSD_TRAIN_DIR = "./dataset/Bank_Checks_Segmentation_Database/TrainSet"
    BCSD_TEST_DIR = "./dataset/Bank_Checks_Segmentation_Database/TestSet"
    src_train, src_test = BCSDDataset(BCSD_TRAIN_DIR), BCSDDataset(BCSD_TEST_DIR)
    train = GeneratorDataset(source=src_train, column_names=["X", "Y"], shuffle=False)
    test = GeneratorDataset(source=src_test, column_names=["X", "Y"], shuffle=False)
    batch_train, batch_test = train.batch(32), test.batch(32)
    iterator_train = batch_train.create_tuple_iterator()
    iterator_test = batch_test.create_tuple_iterator()

    print("TrainSet:")
    for x, y in iterator_train:
        x: mindspore.Tensor; y:mindspore.Tensor
        print(x.shape, y.shape)

    print("TestSet:")
    for x, y in iterator_test:
        x: mindspore.Tensor; y:mindspore.Tensor
        print(x.shape, y.shape)
    