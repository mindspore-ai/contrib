import os
import struct
import requests
import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore.dataset import GeneratorDataset
from mindspore.train import Model, LossMonitor, TimeMonitor
import numpy as np  # 添加这一行，导入 numpy

# 设置环境
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class SimpleNN(nn.Cell):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Flatten layer to convert 28x28 images into 1D vectors
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Dense(28 * 28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def download_file(url, target_path):
    """Download a file from a URL if it doesn't already exist."""
    if not os.path.exists(target_path):
        print(f"Downloading {target_path} from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Check for HTTP errors
            with open(target_path, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download file: {e}")
            raise
    else:
        print(f"{target_path} already exists.")


def load_idx3_ubyte(file_path):
    """
    Load data from IDX3-UBYTE format (MNIST images).
    Returns:
        numpy array of shape (num_images, rows, cols)
    """
    with open(file_path, 'rb') as f:
        # Read the magic number and dimensions
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in file {file_path}")
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return data


def load_idx1_ubyte(file_path):
    """
    Load data from IDX1-UBYTE format (MNIST labels).
    Returns:
        numpy array of shape (num_labels,)
    """
    with open(file_path, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in file {file_path}")
        # Read label data
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(data_path):
    """Load MNIST dataset from local files or download them if missing."""
    print(f"Loading data from {data_path}")

    # Ensure the data directory exists
    os.makedirs(data_path, exist_ok=True)

    # Define file names and their corresponding URLs
    file_urls = {
        'train-images.idx3-ubyte': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train-labels.idx1-ubyte': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        't10k-images.idx3-ubyte': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels.idx1-ubyte': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

    # Download files if they don't exist
    for file_name, file_url in file_urls.items():
        file_path = os.path.join(data_path, file_name)
        download_file(file_url, file_path)

    # Load the data
    train_images = load_idx3_ubyte(os.path.join(data_path, 'train-images.idx3-ubyte')).astype(np.float32) / 255.0  # 归一化
    train_labels = load_idx1_ubyte(os.path.join(data_path, 'train-labels.idx1-ubyte'))
    test_images = load_idx3_ubyte(os.path.join(data_path, 't10k-images.idx3-ubyte')).astype(np.float32) / 255.0  # 归一化
    test_labels = load_idx1_ubyte(os.path.join(data_path, 't10k-labels.idx1-ubyte'))

    def generator(images, labels):
        for img, label in zip(images, labels):
            yield img.reshape(1, 28, 28), int(label)  # 确保数据类型匹配

    train_ds = GeneratorDataset(lambda: generator(train_images, train_labels), column_names=["image", "label"])
    test_ds = GeneratorDataset(lambda: generator(test_images, test_labels), column_names=["image", "label"])

    return train_ds, test_ds


def create_dataset(data_path, batch_size=64):
    """Load and preprocess the MNIST dataset."""
    train_ds, test_ds = load_mnist(data_path)

    # Apply transformations if necessary
    train_ds = train_ds.batch(batch_size=batch_size)
    test_ds = test_ds.batch(batch_size=batch_size)

    return train_ds, test_ds


def main():
    data_path = "./datasets/MNIST"

    # Step 2: Load the MNIST dataset in MindSpore
    train_ds, test_ds = create_dataset(data_path)

    # Step 3: Define the model, loss function, and optimizer
    net = SimpleNN()
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)

    # Step 4: Train the model
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={'accuracy'})
    print("Starting training...")
    model.train(epoch=1, train_dataset=train_ds, callbacks=[LossMonitor(), TimeMonitor()])

    # Step 5: Evaluate the model on the test set
    accuracy = model.eval(test_ds)
    print(f"Test accuracy: {accuracy['accuracy']}")


if __name__ == "__main__":
    main()