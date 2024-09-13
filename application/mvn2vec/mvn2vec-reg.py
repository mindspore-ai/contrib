import mindspore
from mindspore import nn, ops
import gc


class Model_AllCombined(nn.Cell):
    def __init__(self, num_net, len_common_nodes, embedding_dim, embed_freq, batch_size, negative_sampling_size=10):

        super(Model_AllCombined, self).__init__()
        self.n_embedding = len_common_nodes
        self.embed_freq = embed_freq
        self.num_net = num_net
        self.node_embeddings = nn.CellList()
        self.neigh_embeddings = nn.CellList()
        for n_net in range(self.num_net):  # len(G)
            self.node_embeddings.append(
                nn.Embedding(len_common_nodes, embedding_dim))
            self.neigh_embeddings.append(
                nn.Embedding(len_common_nodes, embedding_dim))

        self.negative_sampling_size = negative_sampling_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def construct(self, count, shuffle_indices_nets, nodesidx_nets, neighidx_nets, gamma):
        cost = []
        for i in range(self.num_net):

            batch_indices = shuffle_indices_nets[i][count:count +
                                                    self.batch_size]

            nodesidx = mindspore.Tensor(nodesidx_nets[i][batch_indices])
            node_emb = self.node_embeddings[i](nodesidx).unsqueeze(
                2).view(len(batch_indices), self.embedding_dim, -1)

            sumr1_inner = 0
            for vr in range(self.num_net):
                sumr1_inner += self.node_embeddings[vr](
                    nodesidx).unsqueeze(
                    2).view(len(batch_indices), self.embedding_dim, -1)

            r1 = ops.dist(node_emb, sumr1_inner / self.num_net, p=2) ** 2

            neighsidx = mindspore.Tensor(neighidx_nets[i][batch_indices])
            neigh_emb = self.neigh_embeddings[i](neighsidx).unsqueeze(
                2).view(len(batch_indices), -1, self.embedding_dim)
            loss_positive = ops.logsigmoid(
                ops.bmm(neigh_emb, node_emb)).squeeze().mean()

            sumr2_inner = 0
            for vr in range(self.num_net):
                sumr2_inner += self.neigh_embeddings[vr](neighsidx).unsqueeze(
                    2).view(len(batch_indices), -1, self.embedding_dim)

            r2 = ops.dist(neigh_emb, sumr2_inner / self.num_net, p=2) ** 2

            ############### positive finished##########

            negative_context = self.embed_freq.multinomial(
                len(batch_indices) * neigh_emb.shape[1] *
                self.negative_sampling_size,
                replacement=True)
            # higher freq index higher chance to be sampled
            negative_context_emb = self.neigh_embeddings[i](negative_context).view(len(batch_indices), -1,
                                                                                   self.embedding_dim).neg()

            loss_negative = ops.logsigmoid(ops.bmm(negative_context_emb, node_emb)).squeeze().sum(1).mean(
                0)  # sum through all 10 negative words/columns, row size is batch size, takes average respct to batch size

            sumr2_inner_neg = 0
            for vr in range(self.num_net):
                sumr2_inner_neg += self.neigh_embeddings[vr](
                    negative_context).unsqueeze(
                    2).view(len(batch_indices), -1, self.embedding_dim)
            r2_neg = ops.dist(negative_context_emb,
                              sumr2_inner_neg / self.num_net, p=2) ** 2

            cost.append(
                loss_positive + loss_negative - gamma * (r1 + r2)/self.batch_size - gamma * (r1 + r2_neg)/self.batch_size)  # loss is updated with gamma

            gc.collect()
        return -sum(cost) / len(cost)

    def save_embedding(self, file_name, id2word):
        embeds = self.node_embeddings.weight.data
        fo = open(file_name, 'a+')
        for idx in range(len(embeds)):
            word = id2word[idx]
            embed = ' '.join(embeds[idx])
            fo.write(word + ' ' + embed + '\n')
        fo.close()


if __name__ == '__main__':
    import random

    num_net = 16
    len_common_nodes = 16
    embedding_dim = 4
    embed_freq = ops.randint(
        2, embedding_dim - 1, (num_net, len_common_nodes))
    batch_size = 4
    model = Model_AllCombined(
        num_net, len_common_nodes, embedding_dim, embed_freq, batch_size)
    count = random.randint(0, len_common_nodes - 1)
    shuffle_indices_nets = ops.randint(
        2, len_common_nodes, (num_net, len_common_nodes))
    nodesidx_nets = ops.randint(
        2, embedding_dim - 1, (num_net, len_common_nodes, embedding_dim))
    neighidx_nets = ops.randint(
        2, embedding_dim - 1, (num_net, len_common_nodes, embedding_dim))

    gamma = 0.01
    print(model(count, shuffle_indices_nets, nodesidx_nets, neighidx_nets, gamma))
