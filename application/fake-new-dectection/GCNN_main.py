import numpy as np
import mindspore.dataset as ds
import mindspore as ms
import mindspore.nn as nn
from mindspore_gl.nn import GATConv,AvgPooling
from mindspore_gl import BatchedGraphField
from mindspore import ops
from torch_geometric.datasets import TUDataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import argparse
from main import training_set


parser = argparse.ArgumentParser()

tudataset = TUDataset(root='data/', name='PROTEINS', use_node_attr=True)

parser.add_argument('--seed', type=int, default=777, help='random seed')
# parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--device', type=str, default='cpu', help='specify device')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
args = parser.parse_args()

def graph_data_generator():
    for data in tudataset:
        node_features = data.x.numpy()
        edge_index = data.edge_index.numpy()
        labels = data.y.numpy()
        yield node_features, edge_index, labels

dataset = ds.GeneratorDataset(
    source=graph_data_generator,
    column_names=["node_features", "edge_index", "labels"],
    shuffle=False
)

args.num_classes = 2
args.num_features = 4

train_set, val_set, test_set = dataset.split([0.6,0.1,0.3])
print(len(train_set), len(val_set), len(test_set))


def batched_graph_data_generator(dataset):
    for data in dataset.create_dict_iterator(output_numpy=True):
        node_features = data['node_features']
        edge_index = data['edge_index']
        labels = data['labels']
        n_nodes = node_features.shape[0]
        n_edges = edge_index.shape[1]

        src_idx = ms.Tensor(edge_index[0], ms.int32)
        dst_idx = ms.Tensor(edge_index[1], ms.int32)
        ver_subgraph_idx = ms.Tensor(np.zeros(n_nodes), ms.int32)
        edge_subgraph_idx = ms.Tensor(np.zeros(n_edges), ms.int32)
        graph_mask = ms.Tensor([1], ms.int32)

        batched_graph_field = BatchedGraphField(src_idx, dst_idx, n_nodes, n_edges, ver_subgraph_idx, edge_subgraph_idx,
                                                graph_mask)

        yield batched_graph_field, node_features, labels

train_batched_graph_dataset = ds.GeneratorDataset(
    source=lambda: batched_graph_data_generator(train_set),
    column_names=["batched_graph", "node_features", "labels"],
    shuffle=True
)

val_batched_graph_dataset = ds.GeneratorDataset(
    source=lambda: batched_graph_data_generator(val_set),
    column_names=["batched_graph", "node_features", "labels"],
    shuffle=False
)

test_batched_graph_dataset = ds.GeneratorDataset(
    source=lambda: batched_graph_data_generator(test_set),
    column_names=["batched_graph", "node_features", "labels"],
    shuffle=False
)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.conv1 = GATConv(in_feat_size=self.num_features, out_size=self.nhid, num_attn_head=2)
        self.conv2 = GATConv(in_feat_size=self.nhid * 2, out_size=self.nhid, num_attn_head=2)
        self.fc1 = nn.Dense(in_channels=self.nhid * 2, out_channels=self.nhid)
        self.fc2 = nn.Dense(in_channels=self.nhid, out_channels=args.num_classes)
        self.avg_pooling = AvgPooling()
    def construct(self, x, edge_index, batch):
        x = nn.SeLU()(self.conv1(x, edge_index))
        x = nn.SeLU()(self.conv2(x, edge_index))
        x = self.avg_pooling(x, batch)
        x = nn.SeLU()(x)
        x = self.fc1(x)
        x = nn.Dropout(keep_prob=0.5)(x)
        x = self.fc2(x)
        return nn.LogSoftmax(axis=-1)(x)

def eval(log):
    accuracy, f1_macro, precision, recall = 0, 0, 0, 0
    prob_log, label_log = [], []

    for batch in log:
        pred_y = np.argmax(batch[0].asnumpy(), axis=1)
        y = batch[1].asnumpy().tolist()
        prob_log.extend(batch[0].asnumpy()[:, 1].tolist())
        label_log.extend(y)

        accuracy += accuracy_score(y, pred_y)
        f1_macro += f1_score(y, pred_y, average='macro')
        precision += precision_score(y, pred_y, zero_division=0)
        recall += recall_score(y, pred_y, zero_division=0)

    return accuracy / len(log), f1_macro / len(log), precision / len(log), recall / len(log)

def compute_test(loader, model, loss_fn):
    model.set_train(False)
    loss_test = 0.0
    out_log = []

    for data in loader.create_dict_iterator(output_numpy=True):
        node_features = ms.Tensor(data['node_features'], ms.float32)
        edge_index = data['batched_graph'].get_graph()
        batch = data['batched_graph'].ver_subgraph_idx
        labels = ms.Tensor(data['labels'], ms.int32)

        out = model(node_features, edge_index, batch)
        loss = loss_fn(out, labels)

        out_log.append([ops.softmax(out, axis=1), labels])
        loss_test += loss.asnumpy()

    return eval(out_log), loss_test

model = Net()
loss_fn = nn.NLLLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)

if __name__ == '__main__':
    out_log = []
    model.set_train(True)

    for epoch in tqdm(range(args.epochs)):
        loss_train = 0.0
        for i, data in enumerate(train_batched_graph_dataset.create_dict_iterator(output_numpy=True)):
            node_features = ms.Tensor(data['node_features'], ms.float32)
            edge_index = data['batched_graph'].get_graph()
            batch = data['batched_graph'].ver_subgraph_idx
            labels = ms.Tensor(data['labels'], ms.int32)
            out = model(node_features, edge_index, batch)
            loss = loss_fn(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.asnumpy()

            out_log.append([ops.softmax(out, axis=1), labels])

        acc_train, _, _, recall_train = eval(out_log)

        [acc_val, _, _, recall_val], loss_val = compute_test(val_batched_graph_dataset, model, loss_fn)
        print(f'Epoch {epoch+1}, loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f}, '
              f'recall_train: {recall_train:.4f}, loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f}, '
              f'recall_val: {recall_val:.4f}')

    [acc_test, f1_test, precision_test, recall_test], test_loss = compute_test(test_batched_graph_dataset, model, loss_fn)
    print(f'Test set results: acc: {acc_test:.4f}, f1_macro: {f1_test:.4f}, '
          f'precision: {precision_test:.4f}, recall: {recall_test:.4f}')