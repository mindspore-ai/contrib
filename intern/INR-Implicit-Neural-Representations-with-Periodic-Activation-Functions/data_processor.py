import numpy as np
from PIL import Image

def decode_output_tensor(out_tensor, sidelen):
    return ((out_tensor.view(sidelen, sidelen, 3).asnumpy() * 0.5) + 0.5) * 255

def rescale_image_array(img):
    return ((img / 255 ) - 0.5) / 0.5

def get_input_coords(sidelen, dim = 2, multiple_images = True):
    '''Generates coordinates in range -1 to 1.
    sidelen: int
    dim: int
    multiple_images: bool - True to add an index in {-1,1} if there are 2 images
    '''
    tensors_np = tuple(dim * [np.linspace(-1, 1, num = sidelen)])
    mgrid = np.stack(np.meshgrid(*tensors_np), axis = -1)
    mgrid = np.reshape(mgrid, (-1, dim))
    if multiple_images:
      img_index = np.ones(mgrid.shape[0], np.float32)
      img_index = np.expand_dims(img_index, axis = 1)

      mgrid = np.concatenate([np.concatenate([mgrid, img_index], axis = 1), 
                              np.concatenate([mgrid, -1 * img_index], axis = 1)], axis = 0)
    return mgrid

def get_images_as_tensor(sidelength, img1, img2):
    #reshaping and normalizing in range [-1,1]
    #img1 Resize
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img1_w, img1_h = img1.size
    if img1_w < img1_h:
        img1_h = int(sidelength * (img1_h / img1_w))
        img1_w = sidelength
    else:
        img1_w = int(sidelength * (img1_w / img1_h))
        img1_h = sidelength
    img1 = img1.resize((img1_w, img1_h), Image.BILINEAR)
    #img1 ToTensor
    img1 = np.asarray(img1.convert('RGB'))
    img1 = (img1 / 255.0).astype(np.float32)
    #img1 Normalize
    img1 = (img1 - mean) / std
    img1 = np.transpose(img1, (2, 0, 1))
    
    #img2 Resize
    img2_w, img2_h = img2.size
    if img2_w < img2_h:
        img2_h = int(sidelength * (img2_h / img2_w))
        img2_w = sidelength
    else:
        img2_w = int(sidelength * (img2_w / img2_h))
        img2_h = sidelength
    img2 = img2.resize((img2_w, img2_h), Image.BILINEAR)
    #img2 ToTensor
    img2= np.asarray(img2.convert('RGB'))
    img2 = (img2 / 255.0).astype(np.float32)
    #img2 Normalize
    img2 = (img2 - mean) / std
    img2 = np.transpose(img2, (2, 0, 1))
    
    return img1, img2

class ImagesINR:
    def __init__(self, sidelength, img1, img2):
        img1, img2 = get_images_as_tensor(sidelength, img1, img2)
        img1 = np.reshape(img1, (3, -1))
        img2 = np.reshape(img2, (3, -1))
        self.pixels = np.concatenate([np.transpose(np.reshape(img1, (3, -1)), (1, 0)), 
                                 np.transpose(np.reshape(img2, (3, -1)), (1, 0))], axis = 0)
        self.coords = get_input_coords(sidelength, 2)

    def __len__(self):
        # all the coords are processed during one iteration
        return 1

    def __getitem__(self, idx):    
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels