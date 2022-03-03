import torch
import os
import numpy as np

class GraphData(object):
    """
    Data类需要包含以下基础属性
    (1).edge_index:[source_node,target_node]边关系矩阵;
    　  节点编号从0开始,且节点编号连续;
       支持1).无向图,2).有向图,3).连通图,4).非连通图;
       source_node存储源节点编号,target_node存储目标节点编号;
       数据类型:torch.int64
       形状:[2,edge_num]
    (2).x:节点特征矩阵;
       每行表示一个节点特征向量;
       行号代表节点编号,从0开始且连续;
       数据类型:torch.float32;
       形状:[node_num,node_dim]
    (3).y:节点标签;
       每列表示一个节点的标签;
       列号表示节点编号,从0开始且连续;
       数据类型:torch.int64
       形状:node_num;
    (4).mask矩阵:transductive训练模式数据掩码矩阵
    　　1).self.train_mask
       2).self.val_mask
       3).self.test_mask
       每列表示一个节点掩码;
       列号表示节点编号,从0开始且连续;
       数据类型:torch.uint8
       形状:node_num;
    (5).num_features:特征向量维度;
       数据类型:int
    (6).data_name:数据集名称
    　　数据类型:str
    """
    def __init__(self):

        node_edge_path = os.path.split(os.path.realpath(__file__))[0][:-9] + "/examples/node_edge.txt"
        node_feature_path = os.path.split(os.path.realpath(__file__))[0][:-9] + "/examples/node_feature.txt"
        node_label_path = os.path.split(os.path.realpath(__file__))[0][:-9] + "/examples/node_label.txt"

        # 构建边关系
        source_node = []
        target_node = []
        with open(node_edge_path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                line = line.split(" ")
                source_node.append(int(line[0]))
                target_node.append(int(line[1]))

        edge_index = [source_node, target_node]

        # 构建节点特征
        x = []
        with open(node_feature_path, "r") as f:
            for line in f.readlines():
                x.append(list(eval(line)))

        # 构造节点标签
        y = []
        with open(node_label_path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                line = line.split(" ")
                y.append(int(line[1]))

        # mask向量构造
        y_len = len(y)
        indices = np.arange(y_len).astype('int32')
        idx_train = indices[:int(y_len * 0.5)]
        idx_val = indices[int(y_len * 0.5):int(y_len * 0.5)+int(y_len * 0.25)]
        idx_test = indices[int(y_len * 0.5)+int(y_len * 0.25):]

        train_mask = self.sample_mask(idx_train, y_len)
        val_mask = self.sample_mask(idx_val, y_len)
        test_mask = self.sample_mask(idx_test, y_len)

        # 数据类型转换
        device = torch.device("cuda")

        self.edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)

        self.train_mask = torch.tensor(train_mask, dtype=torch.uint8).to(device)
        self.val_mask = torch.tensor(val_mask, dtype=torch.uint8).to(device)
        self.test_mask = torch.tensor(test_mask, dtype=torch.uint8).to(device)

        self.num_features = int(self.x.shape[1])
        self.data_name = "example_data"

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.int32)

if __name__=="__main__":
    graph = GraphData()
    print(graph)