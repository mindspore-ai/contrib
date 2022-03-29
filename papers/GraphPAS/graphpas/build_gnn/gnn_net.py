import torch
import torch.nn.functional as F
from graphpas.build_gnn.message_passing_net import MessagePassingNet

class GnnNet(torch.nn.Module):

    def __init__(self,
                 architecture,
                 num_feat,
                 layer_num,
                 one_layer_component_num,
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

            if i == 0:
                in_channels = self.num_feat
            else:
                in_channels = out_channels * head_num

            attention_type = self.architecture[i * self.one_layer_component_num + 0]
            aggregator_type = self.architecture[i * self.one_layer_component_num + 1]
            act = self.architecture[i * self.one_layer_component_num + 2]
            head_num = self.architecture[i * self.one_layer_component_num + 3]
            out_channels = self.architecture[i * self.one_layer_component_num + 4]
            concat = True

            if i == self.layer_num - 1 or self.layer_num == 1:
                concat = False

            self.layers.append(MessagePassingNet(in_channels,
                                                 out_channels,
                                                 head_num,
                                                 concat,
                                                 dropout=self.dropout,
                                                 att_type=attention_type,
                                                 agg_type=aggregator_type, ))

            self.acts.append(self.act_map(act))

    def forward(self, x, edge_index_all):
        output = x
        for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
            output = F.dropout(output, p=self.dropout, training=self.training)
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