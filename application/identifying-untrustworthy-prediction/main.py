import os
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindcv.models import create_model
from CSM import cosine_similarity_maps
import numpy as np

context.set_context(mode=context.PYNATIVE_MODE)


def load_cifar10(n_examples=50, data_dir="./cifar-10-batches-bin"):
    dataset = ds.Cifar10Dataset(data_dir, num_samples=n_examples)
    images = []
    labels = []
    for data in dataset.create_dict_iterator():
        images.append(data['image'].asnumpy())
        labels.append(data['label'].asnumpy())
    images = np.array(images).astype(np.float32)

    if images.shape[-1] == 3:
        images = images.transpose(0, 3, 1, 2)

    labels = np.array(labels)
    return Tensor(images), Tensor(labels)


def run_experiment(rows=3, cols=3):
    x, y = load_cifar10(n_examples=50, data_dir="./cifar-10-batches-bin")
    x_noisy = x + Tensor(np.random.rand(*x.shape).astype(np.float32))
    model = create_model(model_name="densenet121", num_classes=10, pretrained=True)
    plot_csm(model, x, y, rows, cols, "CSMs of Clean Data")
    plot_csm(model, x_noisy, y, rows, cols, "CSMs of Noisy Data")


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_csm(model, x, y, rows, cols, title):
    save_path = "./Images/"
    create_directory(save_path)
    csm = cosine_similarity_maps(model, x, True, False)
    csm = csm.asnumpy()
    labels = y.asnumpy()

    fig, ax = plt.subplots(rows, cols)
    total = rows * cols
    for i in range(total):
        r = i // cols
        c = i % cols
        ax[r, c].imshow(csm[i], cmap="coolwarm")
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_title("Label: " + str(labels[i]))
    plt.tight_layout()
    fig.suptitle(title)
    plt.savefig(os.path.join(save_path, "CIFAR10_" + title + ".png"), dpi=800)
    plt.clf()


if __name__ == "__main__":
    run_experiment(rows=2, cols=2)