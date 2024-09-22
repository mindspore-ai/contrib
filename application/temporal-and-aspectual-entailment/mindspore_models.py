import mindspore
from mindspore import nn, Tensor


ACTIVATIONS = {
	'tanh': nn.Tanh,
	'sigmoid': nn.Sigmoid,
	'relu': nn.ReLU,
	'prelu': nn.PReLU,
	'selu': nn.SeLU,
	'elu': nn.ELU
}


class FeedforwardLayer(nn.Cell):
	def __init__(self, input_dim, output_dim, activation, dropout_ratio):
		super().__init__()
		self.feedforward = nn.SequentialCell(
			nn.Dense(input_dim, output_dim),
			ACTIVATIONS[activation](),
			nn.Dropout(p=dropout_ratio)
		)

	def construct(self, sequence):
		return self.feedforward(sequence)


class InflectionFeedforward(nn.Cell):
	def __init__(self, input_dim, num_layers, hidden_size, activations, dropout_ratios):
		super().__init__()
		self.num_layers = num_layers

		if (num_layers > 1 and not isinstance(hidden_size, list)):
			hidden_size = [hidden_size] * num_layers
		if (num_layers > 1 and not isinstance(activations, list)):
			activations = [activations] * num_layers
		if (num_layers > 1 and not isinstance(dropout_ratios, list)):
			dropout_ratios = [dropout_ratios] * num_layers

		self.forward_layer_1 = FeedforwardLayer(input_dim=input_dim, output_dim=hidden_size[0],
												activation=activations[0], dropout_ratio=dropout_ratios[0])

		for i, (size, activation, dropout_ratio) in enumerate(zip(hidden_size[1:], activations[1:], dropout_ratios[1:]), 1):
			layer = FeedforwardLayer(input_dim=hidden_size[i-1], output_dim=input_dim if size == 'XXX' else size,
									 activation=activation, dropout_ratio=dropout_ratio)
			setattr(self, f'forward_layer_{i+1}', layer)

	def construct(self, infinitives):
		for i in range(self.num_layers):
			infinitives = getattr(self, f'forward_layer_{i+1}')(infinitives)

		return infinitives

if __name__ == "__main__":
	infinitives = Tensor([[1, 2, 3], [4, 5, 6]], mindspore.float32)
	ff = InflectionFeedforward(input_dim=3, num_layers=2, hidden_size=[4, 5], activations=['relu', 'tanh'],
							   dropout_ratios=[0.2, 0.3])
	print(ff(infinitives))
