import mindspore as ms
from mindspore import ops, nn, Tensor
import numpy as np

np.random.seed(0)

class layer(nn.Cell):
    def __init__(self):
        super(layer,self).__init__()
        self.linear1 = nn.SequentialCell(
            nn.Dense(1,1024),
            nn.ReLU()
        )
        self.y1 = nn.Dense(1024,1)
        self.y2 = nn.Dense(1024,1)
        self.sigma1 = ms.Parameter(ops.zeros(1))
        self.sigma2 = ms.Parameter(ops.zeros(1))
    def construct(self,X):
        x=self.linear1(X)
        y_1=self.y1(x)
        y_2=self.y2(x)
        return [y_1,y_2],[self.sigma1,self.sigma2]


N = 100
nb_epoch = 5000
batch_size = 20
nb_features = 1024
Q = 1
D1 = 1  # first output
D2 = 1  # second output
def gen_data(N):
    X = np.random.randn(N, Q)
    w1 = 2.
    b1 = 8.
    sigma1 = 1e1  # ground truth
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    w2 = 3
    b2 = 3.
    sigma2 = 1e0  # ground truth
    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2
def mlt(y_pred,y_true,log_vars):
    ys_true = y_pred
    ys_pred = y_true
    loss=0
    for y_true, y_pred, log_var in zip(ys_true, ys_pred, log_vars):
        pre = ops.exp(-log_var)
        loss += ops.sum(pre*(y_true-y_pred)**2+log_var,-1)
    loss = ops.mean(loss)
    return loss
X, Y1, Y2 = gen_data(N)
X = Tensor(X.astype(np.float32))
Y1 = Tensor(Y1.astype(np.float32))
Y2 = Tensor(Y2.astype(np.float32))
model=layer()

optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)
def forward_fn(feat, y_true):
    y_pred, log_vars = model(feat)
    loss = mlt(y_pred, y_true, log_vars)
    return loss, y_pred

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

def train_step(feat, y_true):
    loss, grads = grad_fn(feat, y_true)
    optimizer(grads)
    return loss

for i in range(nb_epoch):
    total_loss = train_step(X, [Y1, Y2])[0]
    if i % 100 == 0:
        print("At epoch %d, Loss: %.4f" % (i, total_loss.asnumpy()))
        print('sigma1: ', model.sigma1.asnumpy())
        print('sigma2: ', model.sigma2.asnumpy())
sigma1 = model.sigma1.asnumpy()
sigma2 = model.sigma2.asnumpy()
print("Estimated sigma1: ", np.exp(sigma1)**0.5)
print("Estimated sigma2: ", np.exp(sigma2)**0.5)
