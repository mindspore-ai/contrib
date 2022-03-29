import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import os
import torch

class CiteNetwork():

    def __init__(self, dataset):
        if dataset in ["cora", "citeseer", "pubmed"]:
            path = os.path.split(os.path.realpath(__file__))[0][:-10] + "data_utils/cite_network/" + dataset
            dataset = Planetoid(path, dataset, T.NormalizeFeatures())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = dataset[0]

            self.edge_index = data.edge_index.to(device)
            self.x = data.x.to(device)
            self.y = data.y.to(device)

            self.train_mask = data.train_mask.to(device)
            self.val_mask = data.val_mask.to(device)
            self.test_mask = data.test_mask.to(device)

            self.num_features = data.num_features
            self.data_name = dataset
        else:
            print("wrong data_name")

if __name__=="__main__":
    data_name = "citeseer"
    graph = CiteNetwork(data_name)
    print(graph)
    pass


