import torch
import torch.nn.functional as F
from graphpas.build_gnn.message_passing_net import MessagePassingNet

class GnnNet(torch.nn.Module):  # 该类使用pytorch需要继承的类

    def __init__(self,
                 architecture,              # 结构组件列表
                 num_feat,                  # 节点特征维度
                 layer_num,                 # GNN层数
                 one_layer_component_num,   # 每层GNN组件数
                 dropout=0.6):
        super(GnnNet, self).__init__()

        self.architecture = architecture
        self.num_feat = num_feat
        self.layer_num = layer_num
        self.one_layer_component_num = one_layer_component_num
        self.dropout = dropout

    def build_architecture(self):
        self.layers = torch.nn.ModuleList()
        self.acts = []

        for i in range(self.layer_num):
            # 初始化GNN输入第一层维度与图特征维度对齐
            if i == 0:
                in_channels = self.num_feat
            else:
                in_channels = out_channels * head_num# 构建第二层GNN输入维度

            # extract layer information 抽取gnn每层的构造信息
            attention_type = self.architecture[i * self.one_layer_component_num + 0]
            aggregator_type = self.architecture[i * self.one_layer_component_num + 1]
            act = self.architecture[i * self.one_layer_component_num + 2]
            head_num = self.architecture[i * self.one_layer_component_num + 3]
            out_channels = self.architecture[i * self.one_layer_component_num + 4]
            concat = True

            if i == self.layer_num - 1 or self.layer_num == 1:
                # 最后一层/单层多头注意力机制下输出的特征向量，维度不再拼接.
                concat = False

            self.layers.append(MessagePassingNet(in_channels,
                                                 out_channels,
                                                 head_num,
                                                 concat,
                                                 dropout=self.dropout,
                                                 att_type=attention_type,
                                                 agg_type=aggregator_type, ))
            """
            # 1.确认节点聚合方式;
            # 2.降维矩阵;
            # 3.多头注意力系数向量a矩阵;
            # 4.构建多头注意力机制偏执bias矩阵;　
            # 5.三个矩阵初始化.
            """
            self.acts.append(self.act_map(act))#构建激活函数对象

    def forward(self, x, edge_index_all):
        output = x #　x:整个图数据的节点特征矩阵, edge_index_all:整个图数据的边关系向量
        for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
            output = F.dropout(output, p=self.dropout, training=self.training)# 对output矩阵实施dropout操作
            output = act(layer(output, edge_index_all))
        return output

    def act_map(self, act):
        if act == "linear":
            return lambda x: x
        elif act == "elu":
            return torch.nn.functional.elu
        elif act == "sigmoid":
            return torch.sigmoid
        elif act == "tanh":
            return torch.tanh
        elif act == "relu":
            return torch.nn.functional.relu
        elif act == "relu6":
            return torch.nn.functional.relu6
        elif act == "softplus":
            return torch.nn.functional.softplus
        elif act == "leaky_relu":
            return torch.nn.functional.leaky_relu
        else:
            raise Exception("wrong activate function:", str(act))