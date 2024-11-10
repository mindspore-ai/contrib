import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor

class ConvAE(nn.Cell):
    def __init__(self):
        super(ConvAE, self).__init__()

        self.initial_layer = nn.SequentialCell(
            nn.Conv1d(1, 8, 30, 2, pad_mode='valid'),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )

        self.Encoder = nn.SequentialCell(
            nn.Conv1d(8, 16, 20, 2, pad_mode='valid'),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, 32, 10, 2, pad_mode='valid'),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Conv1d(32, 64, 10, 1, pad_mode='valid'),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Conv1d(64, 128, 10, 1, pad_mode='valid'),
            nn.BatchNorm1d(128),
            nn.Tanh(),
        )

        self.Decoder = nn.SequentialCell(
            nn.Conv1dTranspose(128, 64, 10, 1, pad_mode='valid'),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Conv1dTranspose(64, 32, 10, 1, pad_mode='valid'),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Conv1dTranspose(32, 16, 10, 2, pad_mode='valid'),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1dTranspose(16, 8, 20, 2, pad_mode='valid'),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Conv1dTranspose(8, 1, 30, 2, pad_mode='valid'),
            nn.Tanh()
        )

        self.linear1 = nn.Dense(720, 720)


    def hybrid_layer(self, x_activity, x_covariate):
        x_activity = x_activity.view(-1, 1, 720)
        first = x_activity.shape[0]
        covariate_weight = ops.randn(first, 4) 
        demo_tensor = x_covariate
        demo_tensor = demo_tensor.squeeze(1)
        print('demo_tensor', demo_tensor.shape)
        print('covariate_weight', covariate_weight.shape)
        res = demo_tensor * covariate_weight
        weight_sum_result = mindspore.Tensor.sum(res)
        self.initial_layer[0].weight.add(weight_sum_result)

        after_x = self.initial_layer(x_activity)

        return after_x

    def initial_layer_m(self, x):
        x_activity = x[:720]
        x_covariate = x[-7:-3]
        x_activity = x_activity.view(-1, 1, 720)
        x_covariate = x_covariate.view(-1, 1, 4)

        after_x = self.hybrid_layer(x_activity, x_covariate)

        return after_x

    def encoder(self, x):
        x_ = self.Encoder(x)
        x_  = x_ .view(-1, 60)
        print('x_ ', x_.shape )
        return x_

    def decoder(self, x):
        x = x.view(-1, 128, 60)
        x = self.Decoder(x)
        print('x', x.shape)
        x = x.view(-1, 720)
        return x

    def construct(self, x):
        x = x.view(-1, 1, len(x))
        after_x = self.initial_layer_m(x)
        encode = self.encoder(after_x)
        decode = self.decoder(encode)
        print(type(decode))
        print('decode', decode.shape)
        
        return decode.view(-1, 720)

# Create the model
model = ConvAE()

# Generate random input, assuming input length is 720
input_data = ops.randn(1, 720)

# Forward pass
output = model(input_data)

# Print the output shape
print("Output shape:", output.shape)
