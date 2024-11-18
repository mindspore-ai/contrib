from PIL import Image
import numpy as np

class PositionDataset:
    """Position encoding dataset"""
    def __init__(self, image_name):
        self.image_name = image_name
        #open routine from fastaiv1
        with open(self.image_name, 'rb') as f:
            self.img = Image.open(f)
            w, h = self.img.size
            neww, newh = int(512 * w / h), 512
            self.width, self.height = neww, newh
            self.img = self.img.resize((neww, newh), Image.BICUBIC)
            self.img = np.asarray(self.img.convert('RGB'))
            self.img = np.transpose(self.img, (1, 0, 2))
            self.img = np.transpose(self.img, (2, 1, 0))
            self.img = (self.img / 255.0).astype(np.float32)
            self.p = None

    def __len__(self):
        return 10 # for now just single image but hardcoding for larger batches

    def __getitem__(self, idx):
        if self.p is None:
            coordh = np.linspace(0, 1, self.height, endpoint = False)
            coordw = np.linspace(0, 1, self.width, endpoint = False)
            self.p = np.stack(np.meshgrid(coordw, coordh), 0).astype(np.float32)
        return self.p, self.img
