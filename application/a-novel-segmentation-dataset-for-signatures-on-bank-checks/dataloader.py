import os
import PIL.Image as Image
import numpy as np

from torch.utils.data.dataset import Dataset


class BCSDDataset(Dataset):

    def __init__(self, path, transforms=None, ext='.jpeg'):
        self.transforms = transforms

        path_x = os.path.join(path, 'X')
        self.images = sorted(os.listdir(path_x))
        self.images = filter(lambda x: x.startswith('X') and x.endswith(ext), self.images)
        self.images = [os.path.join(path_x, im) for im in self.images]

        path_y = os.path.join(path, 'y')
        self.labels = sorted(os.listdir(path_y))
        self.labels = filter(lambda x: x.startswith('y') and x.endswith(ext), self.labels)
        self.labels = [os.path.join(path_y, im) for im in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.images[idx]).convert('RGB').resize((512, 512)))
        y = np.array(Image.open(self.labels[idx]).convert('L').resize((512, 512)))
        y[y!=0] = 1

        if self.transforms is not None:
            x = self.transforms(x)
            y = self.transforms(y)

        return x, y