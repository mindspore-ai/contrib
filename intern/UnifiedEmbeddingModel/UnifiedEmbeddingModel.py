import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from mindspore.common.initializer import Normal
import numpy as np

# Set runtime environment CPU GPU or Ascend
# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

# Hyper-parameters
num_epochs = 100
learning_rate = 0.2
margin = 1.0 # margin in triplet loss
batch_size = 50

class TripletMarginWithDistanceLoss(nn.Cell):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.relu = ops.ReLU()
        self.eps = 1e-8  # 防止除以零
        
    def cosine_similarity(self, x, y):
        """use cosine similarity as distance function"""
        dot_product = ops.ReduceSum()(x * y, axis=1)
        norm_x = ops.Sqrt()(ops.ReduceSum()(x ** 2, axis=1) + self.eps)
        norm_y = ops.Sqrt()(ops.ReduceSum()(y ** 2, axis=1) + self.eps)
        return dot_product / (norm_x * norm_y)
        
    def construct(self, anchor, positive, negative):
        pos_sim = self.cosine_similarity(anchor, positive)
        neg_sim = self.cosine_similarity(anchor, negative)
        losses = self.relu(neg_sim - pos_sim + self.margin)
        return ops.ReduceMean()(losses)

class MyDataset:
    def __init__(self, query, pos, neg):
        self.query = query.asnumpy() if isinstance(query, Tensor) else query
        self.pos = pos.asnumpy() if isinstance(pos, Tensor) else pos
        self.neg = neg.asnumpy() if isinstance(neg, Tensor) else neg
        
    def __getitem__(self, index):
        index = int(index)
        return (
            Tensor(self.query[index]), 
            Tensor(self.pos[index]),
            Tensor(self.neg[index])
        )
    
    def __len__(self):
        return len(self.query)

class QueryEncoder(nn.Cell):
    def __init__(self, input_size):
        super().__init__()
        self.q_fc1 = nn.Dense(input_size, 16, weight_init=Normal(0.02))
        self.q_fc2 = nn.Dense(16, 10, weight_init=Normal(0.02))
        self.q_fc3 = nn.Dense(10, 8, weight_init=Normal(0.02))
        self.relu = ops.ReLU()
        
    def construct(self, x):
        x = self.relu(self.q_fc1(x))
        x = self.relu(self.q_fc2(x))
        x = self.relu(self.q_fc3(x))
        return x
    
class DocumentEncoder(nn.Cell):
    def __init__(self, input_size):
        super().__init__()
        self.d_fc1 = nn.Dense(input_size, 12, weight_init=Normal(0.02))
        self.d_fc2 = nn.Dense(12, 8, weight_init=Normal(0.02))
        self.relu = ops.ReLU()
        
    def construct(self, x):
        x = self.relu(self.d_fc1(x))
        x = self.relu(self.d_fc2(x))
        return x

class Trainer:
    def __init__(self, net_q, net_d, loss_fn, optimizer):
        self.net_q = net_q
        self.net_d = net_d
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_fn = mindspore.value_and_grad(self.forward_fn, None, optimizer.parameters)
        
    def forward_fn(self, query, pos, neg):
        anchor = self.net_q(query)
        positive = self.net_d(pos)
        negative = self.net_d(neg)
        return self.loss_fn(anchor, positive, negative)
    
    def train_step(self, query, pos, neg):
        loss, grads = self.grad_fn(query, pos, neg)
        self.optimizer(grads)
        return loss
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for data in dataset:
                loss = self.train_step(*data)
                total_loss += loss.asnumpy()
            avg_loss = total_loss / dataset.get_dataset_size()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def load_dummy_data(query_size, doc_size):
    query = np.random.randn(100, query_size).astype(np.float32)
    pos = np.random.randn(100, doc_size).astype(np.float32)
    neg = np.random.randn(100, doc_size).astype(np.float32)
    
    return GeneratorDataset(
        MyDataset(query, pos, neg), 
        column_names=["query", "pos", "neg"]
    ).batch(batch_size)

if __name__ == "__main__":
    query_encoder = QueryEncoder(20)
    doc_encoder = DocumentEncoder(15)
    
    params = list(query_encoder.trainable_params()) + list(doc_encoder.trainable_params())
    optimizer = nn.Adam(params, learning_rate=learning_rate)
    
    # use triplet loss
    triplet_loss = TripletMarginWithDistanceLoss(margin=margin)
    
    trainer = Trainer(query_encoder, doc_encoder, triplet_loss, optimizer)
    dataset = load_dummy_data(20, 15)
    trainer.train(dataset, num_epochs)