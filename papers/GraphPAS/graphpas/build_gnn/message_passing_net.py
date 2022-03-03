import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data # for unit test

class MessagePassingNet(MessagePassing):
    """
    包含以下过程:
    1.可训练参数矩阵初始化
    <1>.多头注意力线性变换矩阵 weight 初始化
    <2>.多头注意力节点特征权重向量 att 初始化
    <3>.多头注意力偏执 bias 初始化
    2.消息传递网络前向计算过程
    <1>.去掉图中 edge_index 的自环, 重新为 edge_index 添加自环
    <2>.节点特征向量 x 进行多头注意力线性变换
    <3>.基于atten_type计算多头注意力机制下的attention_weight矩阵
    <4>.attention_weight*x_j计算与权重结合后的节点表示weight_node_representation(其中有部分操作使用dropout)
    <5>.基于{max,mean,add}方式聚合weight_node_representation得到node_representation
    <6>.基于concat标志位,使用concat或mean操作处理node_representation
    """
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

        if agg_type in ["sum"]:  # 确认聚合方式
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

        # Parameter将不可训练的Tensor类型转化为可训练的parameter类型.
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        # 构建特征线性变换矩阵,in_channels输入维度, head*out_channels多头注意力机制输出维度.
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        # 构建多头注意力机制节点特征权重向量a矩阵, heads头数;　
        # 因为节点特征需要拼接后再乘以节点特征注意权重向量a, 所以节点特征注意力权重向量a的维度为self.out_channel*2.
        if bias and concat:# 构建concat多头注意力偏执向量bias
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:# 构建非concat多头注意力bias
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.att_type in ["generalized_linear"]:# 线性注意力
            self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):# 参数矩阵初始化
        glorot(self.weight) # 均匀分布初始化
        glorot(self.att)
        zeros(self.bias)

        if self.att_type in ["generalized_linear"]:
            glorot(self.general_att_layer.weight)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        # 给图移除自环为第二层增加自环准备.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 给图每个节点增加自环, 为聚合做准备
        # prepare
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        # 通过多头线性变换增加节点特征矩阵的个数, 其中每个可学习线性变换矩阵W参数独立.
        # 节点特征矩阵降维, 转化为heads个降维矩阵为多头注意力计算做准备.
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))# 消息传递

    def message(self, x_i, x_j, edge_index, num_nodes):
        # edge_index_j起始节点边关系列表 \ edge_index_i目标节点边关系列表
        if self.att_type == "const":# 已检查正确
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
                # 在训练时, 矩阵中每个元素按照概率p进行随机置零
            weight_node_representation = x_j
        elif self.att_type == "gcn":# 已修改正确
            gcn_weight = self.degree_weight(edge_index)
            weight_node_representation = gcn_weight.view(-1, 1, 1) * x_j
        else:
            weight = self.attention_weight(x_i, x_j)
            # 为图中每一条边计算attention系数weight
            # x_i表示x以edge_index_i列表索引扩展的特征矩阵, x_j表示x以edge_index_j列表索引扩展的特征矩阵
            alpha_weight = softmax(weight, edge_index[0], num_nodes)
            # weight系数在节点的感受域中归一化转换为alpha_weight
            if self.training and self.dropout > 0:
                alpha_weight = F.dropout(alpha_weight, p=self.dropout, training=True)
                # 对注意力权重进行dropout操作
            weight_node_representation = x_j * alpha_weight.view(-1, self.heads, 1)

        return weight_node_representation
        # 结合了权重的节点特征表示

    def degree_weight(self, edge_index):
        row, col = edge_index
        deg = degree(row)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * deg_inv_sqrt[col]

    def attention_weight(self, x_i, x_j):
        if self.att_type == "gat":# 已检查正确
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif self.att_type == "gat_sym":#已检查正确
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

        elif self.att_type == "linear":# 已检查正确
            wl = self.att[:, :, :self.out_channels]  # 将多头节点特征注意力权重向量a矩阵拆分左半部分
            wr = self.att[:, :, self.out_channels:]  # 将多头节点特征注意力权重向量a矩阵拆分右半部分
            al = x_j * wl # al第一维度个由al第二第三维度的矩阵对应元素乘以wl矩阵对应元素
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1) # 按al, ar第二维度, 第三维度矩阵行求和
            alpha = torch.tanh(alpha)
        elif self.att_type == "cos":
            """
            不是标准cos相似度, 没有除以两个内积向量的模.
            因为向量本身太小, 除以模会使得节点特征注意力系数量级过大, 大大影响GNN的性能, 实验验证原因不明.
            """
            # wl = self.att[:, :, :self.out_channels]  # weight left
            # wr = self.att[:, :, self.out_channels:]  # weight right
            # vector_dot = x_i * wl * x_j * wr
            #
            # x_i_vector_dot = x_i * wl * x_i * wl
            # x_i_d_2 = x_i_vector_dot.sum(dim=-1)
            # x_i_d = x_i_d_2.sqrt()
            #
            # x_j_vector_dot = x_j * wr * x_j * wr
            # x_j_d_2 = x_j_vector_dot.sum(dim=-1)
            # x_j_d = x_j_d_2.sqrt()
            #
            # x_i_j_d = x_i_d * x_j_d
            # vector_sum = vector_dot.sum(dim=-1)
            # alpha = vector_sum / x_i_j_d

            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif self.att_type == "generalized_linear":# 已检查正确
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
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
        # aggr_out:按照聚合方式聚合weight_node_representation后的节点特征矩阵
        if self.concat is True:
            node_representation = node_representation.view(-1, self.heads * self.out_channels)
            # 实施拼接操作
        else:
            node_representation = node_representation.mean(dim=1)
            # 实施平均操作

        if self.bias is not None:
            node_representation = node_representation + self.bias

        return node_representation

# unit test
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