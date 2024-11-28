import mindspore
from mindspore import ops, nn, ParameterTuple

class ElasticWeightConsolidation:

    def __init__(self, model, crit, lr = 0.001, weight = 1000000):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate = lr)
        self.mean_params = {}
        self.fisher_params = {}

    def _update_mean_params(self):
        for param in self.model.trainable_params():
            self.mean_params[param.name] = param.copy()

    def _loglikelihood_forward_fn(self, iterator, num_batch):
        log_likelihoods = []
        it = iterator
        batch = 63 if num_batch > 63 else num_batch
        for i in range(batch):
            (input, target) = next(it)
            output = ops.log_softmax(self.model(input), axis = 1)
            log_likelihoods.append(output[:, target])
        log_likelihood = ops.concat(log_likelihoods).mean()
        return log_likelihood

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        current_ds.batch(batch_size)
        iterator = current_ds.create_tuple_iterator()
        grad_logliklihood_fn = mindspore.grad(self._loglikelihood_forward_fn, None, self.model.trainable_params())
        grad_logliklihood = grad_logliklihood_fn(iterator,num_batch)
        for param, grad in zip(self.model.trainable_params(), grad_logliklihood):
            self.fisher_params[param.name] = grad.copy() ** 2

    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param in self.model.trainable_params():
                mean = self.mean_params.get(param.name)
                fisher = self.fisher_params.get(param.name)
                if mean is not None and fisher is not None:
                    losses.append((fisher * (param - mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0
    
    def _forward_fn(self, input, target):
        output = self.model(input)
        loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        return loss

    def forward_backward_update(self, input, target):
        self.model.set_train()
        grad_fn = mindspore.value_and_grad(self._forward_fn, None, weights = self.model.trainable_params())
        loss, grads = grad_fn(input, target)
        self.optimizer(grads)

    def save(self, filename):
        mindspore.save_checkpoint(self.model, filename)

    def load(self, filename):
        mindspore.load_checkpoint(filename, self.model)
        