import mindspore
from mindspore.dataset import GeneratorDataset
from mindspore import nn
from data import PositionDataset
from model import FourierNet
import numpy as np
from PIL import Image

#Inputs
#fourier or not
#num layers
#num units
#image (or images)
#device id
#epochs
#batch_size

def train(
        image,
        num_layers = 4,
        num_units = 256,
        batch_size = 2,
        learning_rate = 1e-3,
        epochs = 250,
        num_workers = 8
        ):
    mindspore.set_context(device_target = "Ascend" if mindspore.hal.is_available("Ascend") else "cpu")
    #create dataset
    loader = PositionDataset(image)
    dataset = GeneratorDataset(source = loader, column_names = ["p","img"], num_parallel_workers = num_workers, shuffle = False)
    dataset = dataset.batch(batch_size)
    #instantiate model
    model = FourierNet(num_layers = num_layers, num_units = num_units)
    optimizer = nn.Adam(model.trainable_params(), learning_rate = learning_rate)
    loss_fn = nn.MSELoss()

    def forward_fn(x, y):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        return loss, y_hat
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux = True)
    
    def train_step(x, y, epoch, i):
        (loss, y_hat), grads = grad_fn(x, y)
        optimizer(grads)
        if epoch % 10 == 0 and i == 0:
            print(f'Epoch: {epoch}. Loss: {loss}')
            y_hat_np = y_hat[0].asnumpy()
            y_hat_np = (y_hat_np * 255.0).astype(np.uint8)
            y_hat_np = np.transpose(y_hat_np, (1, 2, 0))
            image = Image.fromarray(y_hat_np)
            #create folder "image"
            image.save(f'image/test_{epoch}.jpg')

    for epoch in range(epochs):
        for i, batch in enumerate (dataset.create_tuple_iterator()):
            x, y = batch
            train_step(x, y, epoch, i)
            
if __name__ == '__main__':
    #need a image
    train(image = "night.jpg")