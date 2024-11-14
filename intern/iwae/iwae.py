import mindspore
from mindspore import nn, context,Tensor
import mindspore.dataset as ds
import mindspore.ops as ops
import numpy as np

class IWAE(nn.Cell):
    def __init__(self, x_dim=784, h_dim=400):
        super(IWAE, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        # encoder network for computing mean and std of a Gaussian proposal q(h|x)
        self.encoder_base = nn.SequentialCell(
            [nn.Dense(x_dim, 200),
            nn.Tanh(),
            nn.Dense(200, 200),
            nn.Tanh()])
        self.encoder_q_mean = nn.SequentialCell(
            [self.encoder_base, 
            nn.Dense(200, h_dim)])
        self.encoder_q_logvar = nn.SequentialCell(
            [self.encoder_base,
            nn.Dense(200, h_dim)])

        # decoder network for computing mean of a Bernoulli likelihood p(x|h)
        self.decoder_p_mean = nn.SequentialCell(
            [nn.Dense(h_dim, 200),
            nn.Tanh(),
            nn.Dense(200, 200),
            nn.Tanh(),
            nn.Dense(200, x_dim),
            nn.Sigmoid()])

    def construct(self, x, num_samples):
        
        # computing mean and std of Gaussian proposal q(h|x)
        #print("x",x)
        q_mean = self.encoder_q_mean(x)
        #print("q_mean1:",q_mean)
        q_logvar = self.encoder_q_logvar(x)
        q_std = ops.exp(q_logvar / 2)

        # replicating mean and std to generate multiple samples. Unsqueezing to handle batch sizes bigger than 1.
        q_mean = ops.repeat_interleave(q_mean.unsqueeze(1), num_samples, axis=1)
        #print("q_mean1:",q_mean)
        q_std = ops.repeat_interleave(q_std.unsqueeze(1), num_samples, axis=1)

        # generating proposal samples
        # size of h: (batch_size, num_samples, h_size)
        h = q_mean + q_std * ops.randn_like(q_std)
        
         # computing mean of a Bernoulli likelihood p(x|h)
        likelihood_mean = self.decoder_p_mean(h)

        # log p(x|h)
        x = x.unsqueeze(1) # unsqueeze for broadcast

        log_px_given_h = ops.sum(x * ops.log(likelihood_mean) + (1-x) * ops.log(1 - likelihood_mean), dim=-1) # sum over x_dim

        # gaussian prior p(h)
        log_ph = ops.sum(-0.5* ops.log(Tensor(2*np.pi)) - ops.pow(0.5*h,2), dim=-1) # sum over h_dim

        # evaluation of a gaussian proposal q(h|x)
        log_qh_given_x = ops.sum(-0.5* ops.log(Tensor(2*np.pi))-ops.log(q_std) - 0.5*ops.pow((h-q_mean)/q_std, 2), dim=-1) # sum over h_dim
        
        # computing log weights 
        log_w = log_px_given_h + log_ph - log_qh_given_x
       
        # normalized weights through Exp-Normalization trick
        M = ops.max(log_w, axis=-1)[0].unsqueeze(1)
        normalized_w =  ops.exp(log_w - M)/ ops.sum(ops.exp(log_w - M), dim=-1).unsqueeze(1) # unsqueeze for broadcast

        # loss signal        
        loss = ops.sum(ops.stop_gradient(normalized_w) * (log_px_given_h + log_ph - log_qh_given_x), dim=-1) # sum over num_samples
        loss = -ops.mean(loss) # mean over batchs

        # computing log likelihood through Log-Sum-Exp trick
        log_px = M + ops.log((1/num_samples)*ops.sum(ops.exp(log_w - M), dim=-1))  # sum over num_samples
        log_px = ops.mean(log_px) # mean over batches
        
        return loss, log_px, likelihood_mean
 
def main():
    batch_size = 250
    x_dim = 28*28
    h_dim = 50
    num_samples = 5
    num_epochs = 50
    lr = 1e-8
    mindspore.set_seed(114)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

    mnist_dataset_dir = "./data"
    train_dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, usage = "train")
    train_dataset = train_dataset.batch(batch_size).shuffle(len(train_dataset))
    train_iterator = train_dataset.create_tuple_iterator()
    test_dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, usage = "test")
    test_dataset = test_dataset.batch(batch_size)
    test_iterator = test_dataset.create_tuple_iterator()

    model = IWAE(x_dim, h_dim)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    grad_fn = mindspore.value_and_grad(model, None, optimizer.parameters,has_aux=True)
    
    for epoch in range(num_epochs):
        for item in train_iterator:
            
            x = item[0].astype(mindspore.float32).view(batch_size, x_dim)
            (loss, log_px, _),grads = grad_fn(x, num_samples)         
            optimizer(grads)

        print('Epoch [{}/{}],  loss: {:.3f}'.format(epoch + 1, num_epochs, loss.item()))
        print('Epoch [{}/{}],  negative log-likelihood: {:.3f}'.format(epoch + 1, num_epochs, - log_px.item()))
    
    log_px_test = []
    
    for item in test_iterator:
            
        x = item[0].astype(mindspore.float32).view(batch_size, x_dim)
        loss, log_px, _ = model(x, num_samples)           
        log_px_test.append(-log_px.item())
    print('Negative log-likelihood on test set: {:.3f}'.format( ops.mean(Tensor(log_px_test)).item()))

if __name__ == '__main__':
    main()