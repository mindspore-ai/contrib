from examples.graph_data_test import GraphData
from graphpas.build_gnn.graph_sample import GraphSample

data = GraphData()
print("source_edge:", data.edge_index)
print("source_x:", data.x)
print("source_y:", data.y)
sample = GraphSample(data, [1, 7], sample_num=2)
"""
  1.图采样并没有改变原图的拓扑结构,只是在做节点聚合操作时,重新确定中心节点的邻居;
  2.要求采样图包含以下属性
  (1).edge_index:边关系矩阵,数据类型:torch.int64,形状:[2,edge_num];
  (2).x:节点特征矩阵,数据类型:torch.float32,形状:[node_num,node_dim];
  (3).y:节点标签,数据类型:torch.int64,形状:node_num;
  (4).mask:训练模式数据掩码矩阵,数据类型:torch.uint8,形状:node_num;
  3.传参
  (1).graph:图数据类;
  (2).sample_targ_node: 采样目标节点,数据类型:list,元素类型:int,采样目标节点编号范围不能超出原始数据节点编号;
  (3).sample_num: 单个节点采样一阶邻居个数;
  """
data = sample.graph_sample()
print("sample_edge:", data.edge_index)
print("sample_x:", data.x)
print("sample_y:", data.y)