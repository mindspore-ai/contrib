from ue import UnifiedEmbedding
from mindspore import nn, ops, Tensor
import mindspore
import polars as pl
import pathlib

class SimpleNN(nn.Cell):
    def __init__(self, emb_levels, emb_dim, col_map):
        super(SimpleNN, self).__init__()
        self.col_map = col_map
        self.ue = UnifiedEmbedding(emb_levels, emb_dim)
        
        in_dim = sum([len(fnum) for _, fnum in col_map]) * emb_dim
        self.fc1 = nn.Dense(in_dim, 64)
        self.relu = nn.ReLU()
        self.out = nn.Dense(64, 1)
        
    def construct(self, x):
        x_ = list()
        for col, fnum in self.col_map:
            x__ = self.ue(x.select(pl.col(col)).to_numpy().squeeze(), fnum)
            x_.append(x__)
        x_ = ops.concat(x_, axis = 1)
        out = self.fc1(x_)
        out = self.relu(out)
        out = self.out(out)
        return out
    
def load_movielens():
    #NEED MovieLens Dataset
    path = pathlib.Path("ml-1m")

    # UserID::Gender::Age::Occupation::Zip-code
    user_ids = dict()
    with open(path / "users.dat") as f:
        for line in f:
            row = line.strip().split("::")
            user_ids[row[0]] = {"age": row[2], "occupation": row[3], "zip": row[4]}

    # UserID::MovieID::Rating::Timestamp
    ratings = list()
    with open(path / "ratings.dat") as f:
        for line in f:
            row = line.strip().split("::")
            row_ = {"user_id": row[0], 
                    "movie_id": row[1], 
                    "rating": row[2],
                    "timestamp": row[3],
                    "age": user_ids[row[0]]["age"],
                    "occupation": user_ids[row[0]]["occupation"],
                    "zip": user_ids[row[0]]["zip"]}
            ratings.append(row_)
            
    data = pl.DataFrame(ratings).with_columns([
        pl.col("user_id").cast(pl.Int32).alias("user_id_num"), 
        pl.col("timestamp").cast(pl.Int32).alias("ts_num"),
        pl.col("rating").cast(pl.Int32).alias("rating_num")
        ]).filter(pl.col("user_id_num") < 100)
    
    mid = data["ts_num"].median()
    train = data.filter((pl.col("ts_num") < mid))
    val = data.filter(pl.col("ts_num") >= mid)
    train_labels = train.select(pl.col("rating_num")).to_numpy()
    val_labels = val.select(pl.col("rating_num")).to_numpy()
    
    return train, val, train_labels, val_labels

if __name__ == "__main__":

    train, val, train_labels, val_labels = load_movielens()
    
    lr = 0.001
    epochs = 100

    col_map = [("user_id", (0,1)), 
                ("movie_id", (2,3)), 
                ("age", (4,)), 
                ("occupation", (5,)), 
                ("zip", (6,))]

    ffnn = SimpleNN(1000, 10, col_map)

    criterion = nn.MSELoss()
    optimizer = nn.Adam(ffnn.trainable_params(), learning_rate = lr)
    loss_fn = nn.MSELoss()

    def forward_fn(train, train_labels):
        outputs = ffnn(train)
        loss = loss_fn(outputs, train_labels)
        return loss
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)
    
    def train_step(train, train_labels):
        ffnn.set_train()
        loss, grads = grad_fn(train, train_labels)
        optimizer(grads)
        ffnn.set_train(False)
        val_outputs = ffnn(val)
        val_loss = loss_fn(val_outputs, Tensor(val_labels, mindspore.float32))
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    for epoch in range(epochs):
        train_step(train, Tensor(train_labels, mindspore.float32))