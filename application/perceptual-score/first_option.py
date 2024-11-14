import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds


mindspore.set_context(device_target="CPU")


class MultiModalDataset:
    def __init__(self, n_samples: int = 1000, first_sample_size: int = 5, second_sample_size: int = 7):
        self.n_samples = n_samples
        self.first_sample_size = first_sample_size
        self.second_sample_size = second_sample_size

        self.first = np.random.randn(self.n_samples, self.first_sample_size).astype(np.float32)
        self.second = np.random.rand(self.n_samples, self.second_sample_size).astype(np.float32)
        self.labels = np.random.randint(0, 2, (self.n_samples,)).astype(np.int32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.first[idx], self.second[idx], self.labels[idx]
    

class TwoLayersFCNetwork(nn.Cell):
    def __init__(self, input_size: int = 12, mid_dim: int = 7, output_size: int = 2):
        super(TwoLayersFCNetwork, self).__init__()
        self.network = nn.SequentialCell(
            nn.Dense(input_size, mid_dim),
            nn.ReLU(),
            nn.Dense(mid_dim, output_size)
        )

    def construct(self, x: Tensor, y: Tensor) -> Tensor:

        combined = ops.Concat(1)([x, y])
        combined = self.network(combined)
        return combined


def evaluate_risk(data_loader: ds.GeneratorDataset, model: nn.Cell, loss_fn: nn.Cell, permute_first: bool = False) -> float:
    running_loss = 0.0
    data_points = 0

    for first_modality, second_modality, labels in data_loader:
        if permute_first:
            first_modality = first_modality[Tensor(np.random.permutation(first_modality.shape[0]))]

        outputs = model(first_modality, second_modality)
        
        loss = loss_fn(outputs, labels).sum()

        
        running_loss += loss.asnumpy()
        data_points += first_modality.shape[0]

    return running_loss / data_points

if __name__ == '__main__':
    n_samples: int = 1000
    first_sample_size: int = 5
    second_sample_size: int = 7

    model = TwoLayersFCNetwork(input_size=first_sample_size + second_sample_size)

    dataset = MultiModalDataset(n_samples=n_samples,
                                first_sample_size=first_sample_size,
                                second_sample_size=second_sample_size)
    train_dataloader = ds.GeneratorDataset(source=dataset, column_names=["first", "second", "labels"], shuffle=True)
    train_dataloader = train_dataloader.batch(batch_size=64)

    loss_fn = nn.CrossEntropyLoss()

    risk = evaluate_risk(data_loader=train_dataloader,
                         model=model,
                         loss_fn=loss_fn,
                         permute_first=True)
    
    permuted_risk = evaluate_risk(data_loader=train_dataset,
                                  model=model,
                                  loss_fn=loss_fn,
                                  permute_first=True)

    print(risk - permuted_risk)