import mindspore
from mindspore import Tensor, ops

import numpy as np


def global_add_pool(x, batch, batch_size):
    """
    对每个图的节点特征进行全局加和池化。
    
    参数:
    - x: 节点特征，形状为 [num_nodes, feature_dim]
    - batch: 每个节点对应的图的批次索引，形状为 [num_nodes]
    - batch_size: 批次中图的数量
    
    返回:
    - 输出张量，形状为 [batch_size, feature_dim]，表示每个图的节点特征加和。
    """
    feature_dim = x.shape[1]  # 特征维度
    out = ops.Zeros()((batch_size, feature_dim), mindspore.float32)  # 初始化输出张量
    for i in range(batch_size):
        # 对每个批次图中的节点特征进行加和
        mask = batch == i  # 选择属于第 i 个图的节点
        out[i] = ops.ReduceSum()(x[mask], 0)  # 按节点加和
    return out


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    inv_mask = ~mask
    loop_weight = ops.full(
        (num_nodes, ),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
)

    if edge_weight is not None:
        #print(edge_index)
        assert edge_weight.shape[0] == edge_index.shape[1]
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.shape[0] > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = ops.cat([edge_weight[mask], loop_weight], axis=0)

    loop_index = ops.arange(0, num_nodes, dtype=row.dtype)
    loop_index = loop_index.unsqueeze(0).tile((2, 1))
    edge_index = ops.cat([edge_index[:, mask], loop_index], axis=1)

    return edge_index, edge_weight


class NewSGConv(mindspore.nn.Cell):
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__()
        self.cached = cached
        self.K = K
        self.cached_result = Tensor([num_features, num_classes], dtype=mindspore.float32)
        self.lin = mindspore.nn.Dense(num_features, num_classes, has_bias=bias)
    
    # allow negative edge weights
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = ops.ones((edge_index.shape[1], ), dtype=dtype)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        abs_edge_weight = ops.Abs()(edge_weight)
        deg = mindspore.Parameter(Tensor(np.zeros((num_nodes,), dtype=np.float32)))
        deg = ops.tensor_scatter_elements(input_x = deg, indices = row, updates = abs_edge_weight, axis = 0)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def construct(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached :
            edge_index, norm = NewSGConv.norm(
                edge_index, x.shape[0], edge_weight, dtype=x.dtype)
            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x
        return self.lin(self.cached_result)

    def propagate(self, edge_index, x, norm):
        """
        Propagate step, consists of message, aggregate, and update.
        :param edge_index: Edge indices (2, num_edges)
        :param x: Node features
        :param norm: Normalization factor for edges
        """
        # Step 1: Message
        x_j = self.message(edge_index, x, norm)
        # Step 2: Aggregate
        aggr = self.aggregate(edge_index,x_j,x)

        # Step 3: Update (Apply aggregation to nodes)
        x = self.update(aggr, x)
        return x

    def message(self, edge_index, x, norm):
        """
        Message passing step.
        :param edge_index: Edge indices (2, num_edges)
        :param x: Node features
        :param norm: Normalization factor for edges
        """
        source, target = edge_index
        x_j = x[source]  
        return norm.view(-1, 1) * x_j
    def aggregate(self,edge_index, x_j,x):
        """
        Aggregation step. 
        """
        source, target = edge_index
        aggr = ops.zeros_like(x)
        for i in range(x_j.shape[0]):
            aggr[target[i]] += x_j[i]
        return aggr
    def update(self, aggr, x):
        """
        Update step. 
        """
        return x+aggr  



class ReverseLayerF(mindspore.nn.Cell):
    def __init__(self, alpha):
        super(ReverseLayerF, self).__init__()
        self.alpha = alpha

    def construct(self, x):
        return x

    def bprop(self, input , output,grad_output):
        # 反向传播，类似 PyTorch 中的 grad_output.neg() * alpha
        return (grad_output * (-self.alpha),)


class SymSimGCNNet(mindspore.nn.Cell):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens,alpha, num_classes, K, dropout=0.5, domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions
            num_classes: number of emotion classes
            K: number of layers
            dropout: dropout rate in final linear layer
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = ops.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        self.edge_weight = mindspore.Parameter(edge_weight, requires_grad=learn_edge_weight)
        self.dropout = dropout
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)
        self.fc = mindspore.nn.Dense(num_hiddens[0], num_classes)
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = mindspore.nn.Dense(num_hiddens[0], 2)
        self.reverselayerF = ReverseLayerF(alpha)

    def construct(self, data, alpha=0):
        batch_size = len(data['y'])
        x, edge_index = data['x'], data['edge_index']
        edge_weight = ops.zeros((self.num_nodes, self.num_nodes))
        edge_weight[self.xs, self.ys] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - ops.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = ops.relu(self.conv1(x, edge_index, edge_weight))
        
        # domain classification
        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = self.reverselayerF(x)
            domain_output = self.domain_classifier(reverse_x)
        x = global_add_pool(x, data['batch'], batch_size=batch_size)
        x = ops.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x, domain_output
    
