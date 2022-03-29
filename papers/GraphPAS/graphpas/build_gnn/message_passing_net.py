import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class MessagePassingNet(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 att_type="gat",
                 agg_type="sum"):

        if agg_type in ["sum"]:
            super(MessagePassingNet, self).__init__('add')
        elif agg_type in ["mean", "max"]:
            super(MessagePassingNet, self).__init__(agg_type)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.att_type in ["generalized_linear"]:
            self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        if self.att_type in ["generalized_linear"]:
            glorot(self.general_att_layer.weight)

    def forward(self, x_, edge_index_):
        edge_index__, _ = remove_self_loops(edge_index_)

        edge_index, _ = add_self_loops(edge_index__, num_nodes=x_.size(0))
        x = torch.mm(x_, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index_, num_nodes):

        if self.att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            weight_node_representation = x_j
        elif self.att_type == "gcn":
            gcn_weight = self.degree_weight(edge_index_)
            weight_node_representation = gcn_weight.view(-1, 1, 1) * x_j
        else:
            weight = self.attention_weight(x_i, x_j)
            alpha_weight = softmax(weight, edge_index_[0], num_nodes)
            if self.training and self.dropout > 0:
                alpha_weight = F.dropout(alpha_weight, p=self.dropout, training=True)

            weight_node_representation = x_j * alpha_weight.view(-1, self.heads, 1)

        return weight_node_representation


    def degree_weight(self, edge_index_):
        row, col = edge_index_
        deg = degree(row)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * deg_inv_sqrt[col]

    def attention_weight(self, x_i, x_j):
        if self.att_type == "gat":
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif self.att_type == "gat_sym":
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

        elif self.att_type == "linear":
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            al = x_j * wl
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1)
            alpha = torch.tanh(alpha)
        elif self.att_type == "cos":
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif self.att_type == "generalized_linear":
            wl = self.att[:, :, :self.out_channels]
            wr = self.att[:, :, self.out_channels:]
            al = x_i * wl
            ar = x_j * wr
            alpha = al + ar
            alpha = torch.tanh(alpha)
            alpha = self.general_att_layer(alpha)
        else:
            raise Exception("Wrong attention type:", self.att_type)
        return alpha

    def update(self, aggr_out):
        node_representation = aggr_out

        if self.concat is True:
            node_representation = node_representation.view(-1, self.heads * self.out_channels)
        else:
            node_representation = node_representation.mean(dim=1)
        if self.bias is not None:
            node_representation = node_representation + self.bias

        return node_representation

if __name__=="__main__":
    edges = [[0, 0, 0, 1, 2, 2, 3, 3], [1, 2, 3, 0, 0, 3, 0, 2]]
    edge_index = torch.tensor(edges, dtype=torch.long)

    node_features = [[-1, 1, 2], [1, 1, 1], [0, 1, 2], [3, 1, 2]]
    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    x = data.x
    edge_index = data.edge_index

    GNN = MessagePassingNet(3, 5)

    x = GNN(x, edge_index)

    print(x)