import time
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, Tensor
from mindspore.dataset.transforms import Compose
from mindspore.common.initializer import HeUniform
from mindspore.dataset.vision import ToTensor, Normalize


def one_hot_encode(img0, lab):
    img = img0.copy()
    img[:, :10] = img0.min()
    img[range(img0.shape[0]), lab] = img0.max()
    return img

def main():
    #Load MNIST Data
    data_path = './MNIST_data/'

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        lambda x: x.flatten()
    ])

    train_loader = ds.MnistDataset(data_path, usage='train')
    train_loader = train_loader.map(operations=transform, input_columns="image")
    train_loader = train_loader.batch(60000)

    test_loader = ds.MnistDataset(data_path, usage='test')
    test_loader = test_loader.map(operations=transform, input_columns="image")
    test_loader = test_loader.batch(10000)

    
    ms.context.set_context(device_target="GPU")
    print('Using device:', ms.context.get_context('device_target'))
    device = ms.context.get_context('device_target')

    for data in train_loader.create_dict_iterator():
        img0 = data["image"].asnumpy()
        lab = data["label"].asnumpy()
        break

    for data in test_loader.create_dict_iterator():
        img0_tst = data["image"].asnumpy()
        lab_tst = data["label"].asnumpy()
        break

    # Forward Forward Applied to a Single Perceptron for MNIST Classification
    n_input, n_out = 784, 125
    batch_size, learning_rate = 10, 0.0003
    g_threshold = 10
    epochs = 250

    perceptron = nn.SequentialCell(
        nn.Dense(n_input, n_out, weight_init=HeUniform(), has_bias=True),
        nn.ReLU()
    )

    optimizer = nn.Adam(perceptron.trainable_params(), learning_rate=learning_rate)

    # Define forward propagation and loss calculations
    def forward_fn(img_pos_batch, img_neg_batch):
        g_pos = (perceptron(img_pos_batch)**2).mean(axis=1)
        loss_pos = ops.log(1 + ops.exp(-(g_pos - g_threshold))).sum()

        g_neg = (perceptron(img_neg_batch)**2).mean(axis=1)
        loss_neg = ops.log(1 + ops.exp(g_neg - g_threshold)).sum()

        loss = loss_pos + loss_neg
        return loss

    # Define the gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    N_trn = img0.shape[0] # Use all training images (60000)

    tic = time.time()

    for epoch in range(epochs):
        img = img0.copy()

        for i in range(10):  # Random jittering of training images up to 2 pixels
            dx, dy = ops.randint(-2, 2, (2,))
            dx, dy = int(dx.asnumpy()), int(dy.asnumpy())

            # Scroll the rows and columns separately
            img[i] = ops.roll(Tensor(img0[i].reshape(28, 28)), shifts=dy, dims=0).flatten().asnumpy()
            img[i] = ops.roll(Tensor(img[i].reshape(28, 28)), shifts=dx, dims=1).flatten().asnumpy()

        perm = np.random.permutation(N_trn)
        img_pos = one_hot_encode(img[perm], lab[perm])

        lab_permuted = Tensor(lab[perm], dtype=ms.int32)
        rand_integers = ops.randint(low=1, high=10, size=(lab_permuted.shape))
        lab_neg = ops.add(lab_permuted, rand_integers)
        lab_neg = ops.select(ops.greater(lab_neg, 9), ops.sub(lab_neg, 10), lab_neg).asnumpy()
        img_neg = one_hot_encode(img[perm], lab_neg)  # Bad data (random error in label)

        L_tot = 0

        for i in range(0, N_trn, batch_size):
            perceptron.set_train(True)

            # Goodness and loss for good data in batch
            img_pos_batch = img_pos[i:i+batch_size]
            img_pos_batch = Tensor(img_pos_batch, dtype=ms.float32)

            # Goodness and loss for bad data in batch
            img_neg_batch = img_neg[i:i+batch_size]
            img_neg_batch = Tensor(img_neg_batch, dtype=ms.float32)

            loss = forward_fn(img_pos_batch, img_neg_batch)
            L_tot += loss.asnumpy() # Accumulate total loss for epoch

            grads = grad_fn(img_pos_batch, img_neg_batch)[1] # Compute gradients
            optimizer(grads) # Update parameters

        # Test model with validation set
        N_tst = img0_tst.shape[0]  # Use all test images (10000)
        
        # Evaluate goodness for all test images and labels 0...9
        g_tst = ops.zeros((10, N_tst), dtype=ms.float32)
        for n in range(10):
            img_tst = one_hot_encode(img0_tst, n * np.ones_like(lab_tst))
            img_tst = Tensor(img_tst, dtype=ms.float32)
            g_tst[n] = ((perceptron(img_tst)**2).mean(axis=1))
        predicted_label = g_tst.argmax(axis=0)
        
        # Count number of correctly classified images in validation set
        correct_predictions = predicted_label == Tensor(lab_tst, dtype=ms.int32)
        Ncorrect = correct_predictions.sum().asnumpy()

        print(f"Epoch {epoch+1}:\tLoss {L_tot}\tTime {round(time.time() - tic)}s\tTest Error {100 - Ncorrect / N_tst * 100}%")

if __name__ == "__main__":
    main()