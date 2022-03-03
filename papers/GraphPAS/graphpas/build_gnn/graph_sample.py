import torch
import random
import numpy as np
from examples.graph_data_test import GraphData # for unit test

class GraphSample(object):
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
    def __init__(self, graph, sample_target_node, sample_num=5):
        self.data = graph
        self.sample_num = sample_num
        self.sample_target_node = sample_target_node

    def graph_sample(self):

        # 采样边关系构造
#=========================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_target_node = self.sample_target_node  # 采样的目标节点
        edge_index = self.data.edge_index  # 原始图边关系

        edge_index_j, edge_index_i = edge_index.to("cpu")
        edge_index_i = edge_index_i.numpy().tolist()
        edge_index_j = edge_index_j.numpy().tolist()

        s_node = []  # 采样后的原始节点索引
        t_node = []  # 采样后的目标节点索引

        for node_index in sample_target_node:

            neighbor_t_node = []

            # 找采样目标节点的邻居节点id集合
            for index in range(len(edge_index_i)):
                if node_index == edge_index_i[index]:
                    neighbor_t_node.append(edge_index_j[index])
            # 采样
            if self.sample_num <= len(neighbor_t_node):
                neighbor_t_node = random.sample(neighbor_t_node, k=self.sample_num)
            else:
                neighbor_t_node = random.choices(neighbor_t_node, k=self.sample_num)
            # 构造temp_t_node
            temp_t_node = [node_index for i in range(len(neighbor_t_node))]

            # 更新s_node,t_node
            s_node = s_node + neighbor_t_node
            t_node = t_node + temp_t_node

        # 重新给采样后的图数据中节点编号
        old_node_id = s_node + t_node  # 合并
        old_node_id.sort()  # 排序
        new_node_id = []  # 新建节点编号列表
        node_id = 0  # 新建节点编号从0开始
        temp_id = old_node_id[0]  # 获取原始排序节点id第一个值

        # 构建映射, 原始节点id与新节点id映射关系
        for s_id in old_node_id:
            if s_id != temp_id:
                node_id += 1
                new_node_id.append(node_id)
                temp_id = s_id
            else:
                new_node_id.append(node_id)

        # 基于映射关系重新编号
        s_node_ = []
        t_node_ = []
        for node in s_node:
            index = old_node_id.index(node)
            s_node_.append(new_node_id[index])
        for node in t_node:
            index = old_node_id.index(node)
            t_node_.append(new_node_id[index])

        # 边补充调整,保证采样后的图是无向图
        s_node, t_node = self.edge_supplement(s_node_, t_node_)
        edge_index = [s_node, t_node]

        # 重新确定图的节点规模
        node_num = []
        for node in s_node:
            if node not in node_num:
                node_num.append(node)
        self.data.node_num = len(node_num)

        # gpu使用
        edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)
        self.data.edge_index = edge_index
#=========================================================================

        # 对节点特征矩阵进行重构, 基于old_node_id
        if self.data.x is not None:
            x = self.data.x.to("cpu")
            temp_id = old_node_id[0]
            x_ = x[temp_id].view(-1, len(x[temp_id]))
            for node_id in old_node_id:
                if node_id == temp_id:
                    continue
                else:
                    x_ = torch.cat((x_, x[node_id].view(-1, len(x[node_id]))), 0)
                    temp_id = node_id
            # 加载到gpu上
            self.data.x = x_.to(device)
        else:
            pass
#==========================================================================

        # 对节点标签tensor进行重构,基于old_node_id
        rebuild_mask = False
        if self.data.y is not None:
            y = self.data.y.to("cpu")
            temp_id = old_node_id[0]
            y_ = y[temp_id].view(-1)
            for node_id in old_node_id:
                if node_id == temp_id:
                    continue
                else:
                    y_ = torch.cat((y_, y[node_id].view(-1)), 0)
                    temp_id = node_id
            if len(y_) < len(y):
                rebuild_mask = True
            # 加载到gpu上
            self.data.y = y_.to(device)
        else:
            pass
#==========================================================================
        # 重构 mask
        if rebuild_mask and (self.data.train_mask is not None):

            train_mask = self.data.train_mask.to("cpu")
            val_mask = self.data.val_mask.to("cpu")
            test_mask = self.data.test_mask.to("cpu")

            mask = 0
            for i in train_mask:
                if i:
                    mask += 1
            percentage_train_mask = int(mask / len(train_mask))

            mask = 0
            for i in val_mask:
                if i:
                    mask += 1
            percentage_val_mask = int(mask / len(val_mask))

            mask = 0
            for i in test_mask:
                if i:
                    mask += 1
            percentage_test_mask = int(mask / len(test_mask))

            y_len = len(y_)
            indices = np.arange(y_len).astype('int32')
            idx_train = indices[:int(y_len * percentage_train_mask)]
            idx_val = indices[int(y_len * percentage_train_mask):
                              int(y_len * percentage_train_mask) + int(y_len * percentage_val_mask)]
            idx_test = indices[int(y_len * percentage_train_mask) + int(y_len * percentage_val_mask):
                               int(y_len * percentage_train_mask) + int(y_len * percentage_val_mask) +
                               int(percentage_test_mask)]

            train_mask = self.sample_mask(idx_train, y_len)
            val_mask = self.sample_mask(idx_val, y_len)
            test_mask = self.sample_mask(idx_test, y_len)

            train_mask = torch.tensor(train_mask, dtype=torch.uint8).to(device)
            val_mask = torch.tensor(val_mask, dtype=torch.uint8).to(device)
            test_mask = torch.tensor(test_mask, dtype=torch.uint8).to(device)

            self.data.train_mask = train_mask
            self.data.val_mask = val_mask
            self.data.test_mask = test_mask

        else:
            pass
# =======================================================================

        return self.data

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.int32)

    def edge_supplement(self, s_node, t_node):

        history = []
        add_edge = []

        for index in range(len(s_node)):

            node_id_j = s_node[index]
            node_id_i = t_node[index]

            if [node_id_j, node_id_i] in history:
                continue

            temp_ji = []
            temp_ij = []

            temp_ji.append([node_id_j, node_id_i])

            for idx in range(index + 1, len(s_node)):

                if node_id_j == s_node[idx]:
                    if node_id_i == t_node[idx]:
                        temp_ji.append([node_id_j, node_id_i])

                if node_id_i == s_node[idx]:
                    if node_id_j == t_node[idx]:
                        temp_ij.append([node_id_i, node_id_j])

            # 这对边的关系正确不需要补充
            if len(temp_ji) == len(temp_ij):
                history.append([node_id_j, node_id_i])
                history.append([node_id_i, node_id_j])
                continue

            # temp_ji大于temp_ij,补充temp_ij关系
            if len(temp_ji) > len(temp_ij):
                for i in range(len(temp_ji) - len(temp_ij)):
                    add_edge.append([node_id_i, node_id_j])
                    history.append([node_id_j, node_id_i])
                    history.append([node_id_i, node_id_j])
                    continue

            # temp_ij大于temp_ji,补充temp_ji关系
            if len(temp_ji) < len(temp_ij):
                for i in range(len(temp_ij) - len(temp_ji)):
                    add_edge.append([node_id_j, node_id_i])
                    history.append([node_id_j, node_id_i])
                    history.append([node_id_i, node_id_j])

        for node_ij in add_edge:
            s_node.append(node_ij[0])
            t_node.append(node_ij[1])

        return s_node, t_node

if __name__ == "__main__":
    data = GraphData()
    print("source_edge:", data.edge_index)
    print("source_x:", data.x)
    print("source_y:", data.y)
    sample = GraphSample(data, [1, 7], sample_num=2)
    data = sample.graph_sample()
    print("sample_edge:", data.edge_index)
    print("sample_x:", data.x)
    print("sample_y:", data.y)

