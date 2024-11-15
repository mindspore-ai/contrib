'''
This File contains the VAE architecture used for majority of the testing.
'''
import mindspore
from mindspore import nn, ops, Tensor
import numpy as np

class VariationalEncoder(nn.Cell):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.encoder_conv = nn.SequentialCell([
            nn.Conv2d(1, 16, 5, stride = 2, pad_mode = 'pad', padding = 1, has_bias = True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2d(16, 32, 5, stride=2, pad_mode = 'pad', padding = 1, has_bias = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2d(32, 64, 5, stride=2, pad_mode = 'pad', padding = 0, has_bias = True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2d(64, 128, 5, stride=2, pad_mode = 'pad', padding = 0, has_bias = True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2d(128, 256, 5, stride=2, pad_mode = 'pad', padding = 0, has_bias = True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2d(256, 512, 5, stride=2, pad_mode = 'pad', padding = 0, has_bias = True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2d(512, 1024, 5, stride=2, pad_mode = 'pad', padding = 0, has_bias = True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(alpha = 0.01)
        ])

        self.linear1 = nn.Dense(1 * 1 * 1024, latent_dims)
        self.linear2 = nn.Dense(latent_dims, latent_dims)
        self.linear3 = nn.Dense(latent_dims, latent_dims)

        # initialise non model parameters
        self.N = nn.probability.distribution.Normal(0, 1)
        self.kl = 0  # stores KL divergence

        self.mu = None
        self.sigma = None

    def construct(self, x):
        x = self.encoder_conv(x)
        x = ops.flatten(x, start_dim = 1)
        x = ops.leaky_relu(self.linear1(x), alpha = 0.01)
        mu = self.linear2(x)
        sigma = ops.exp(self.linear3(x))

        self.mu = mu
        self.sigma = sigma

        z = mu + sigma * self.N.sample(mu.shape)
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
        # these help make sense of below, although interested in a different method for computing KL divergence
        self.kl = (sigma ** 2 + mu ** 2 - ops.log(sigma) - 1 / 2).sum()

        return z


class VariationalDecoder(nn.Cell):
    def __init__(self, latent_dims):
        super(VariationalDecoder, self).__init__()

        self.decoder_lin = nn.SequentialCell([
            nn.Dense(latent_dims, latent_dims),
            nn.LeakyReLU(alpha = 0.01),
            nn.Dense(latent_dims, 1 * 1* 1024),
            nn.LeakyReLU(alpha = 0.01),
        ])

        # self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1024, 1, 1))

        self.decoder_conv = nn.SequentialCell([
            nn.Conv2dTranspose(1024, 512, 5, stride=2, pad_mode = 'valid', has_bias = True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2dTranspose(512, 256, 5, stride=2, pad_mode = 'valid', has_bias = True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2dTranspose(256, 128, 5, stride=2, pad_mode = 'valid', has_bias = True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2dTranspose(128, 64, 5, stride=2, pad_mode = 'valid', has_bias = True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2dTranspose(64, 32, 5, stride=2, pad_mode = 'valid', has_bias = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2dTranspose(32, 16, 5, stride=2, pad_mode = 'valid', has_bias = True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(alpha = 0.01),
            nn.Conv2dTranspose(16, 1, 5, stride=2, pad_mode = 'valid', has_bias = True)
        ])

    def construct(self, x):
        x = self.decoder_lin(x)
        # x = self.unflatten(x)
        x = ops.reshape(x, (-1, 1024, 1, 1)) # i think 1024 is an error
        x = self.decoder_conv(x)
        x = ops.sigmoid(x)
        return x

class VariationalAutoencoder(nn.Cell):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = VariationalDecoder(latent_dims)
        self.latent_dims = latent_dims

    def construct(self, x):
        z = self.encoder(x)
        return self.decoder(z)

if __name__ == "__main__":
    model = VariationalAutoencoder(512)
    batch_size = 128
    input = Tensor(np.random.randn(batch_size, 1, 512, 512).astype(np.float32))
    output = model(input)
    print("test done")
    print(output)