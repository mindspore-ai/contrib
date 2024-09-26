import random
import mindspore as ms
from mindspore import ops

random.seed(0)
ms.manual_seed(0)
ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")

class MTPLoss:
    def __init__(self,
                 modes=3,
                 prediction_steps=16,
                 alpha_class=0.1
                 ):
        self.modes = modes
        self.prediction_steps = prediction_steps
        self.mode_size = self._get_mode_size()
        self.alpha_class = alpha_class

    def get_output_size(self):
        return (self.prediction_steps + 1) * self.modes

    def _get_mode_size(self):
        return self.prediction_steps + 1

    def my_dist(self, outputs, data_flat):
        # shape to the same multi-dim
        data = data_flat.view(-1, 1, self.mode_size - 1)

        # using all but the probability
        prediction = outputs[:, :, 0:self.mode_size - 1]
        unscaled_norm = ops.norm(prediction - data, ord=1, dim=2)
        return unscaled_norm / data_flat.shape[1]

    def _expand(self, output):
        return output.view(-1, self.modes, self.mode_size)

    def __call__(self, output, expected_inversion, my_arange):
        """
        The loss between the network output and the expected output
        :param output:               the output of the network that contains predictions and probabilities
        :param expected_inversion:   the inversion expected from the data point
        :param my_arange:            a sequential vector to avoid recomputation
        :return:
        """
        expanded = self._expand(output)
        data_flat = expected_inversion.view(-1, self.mode_size - 1)
        dists = self.my_dist(expanded, data_flat)
        best_mode = ops.argmin(dists, 1)
        # last index is reserved for probability
        prob_raw = expanded[:, :, self.mode_size - 1]
        log_prob = ops.log_softmax(prob_raw, axis=1)
        prob_contrib = -log_prob[my_arange, best_mode]
        # the log of the probability (non-normalized)
        norm_contrib = dists[my_arange, best_mode]
        # the norm contribution
        result = self.alpha_class * prob_contrib + norm_contrib
        return result


if __name__ == '__main__':
    batch_len = 1
    modes = 7
    prediction_steps = 32
    mtp_loss = MTPLoss(modes=modes, prediction_steps=prediction_steps)

    # creating some reference data
    expected_inversion = (ops.arange(32) * 0.01).reshape((1, -1))

    # my_arange is an auxiliary variable which is the size of batch len
    my_arange = ops.arange(batch_len)

    # creating some output in the shape expected from the neural network
    steps_with_probaility = prediction_steps + 1
    expected_inversion_with_probability = ops.ones((batch_len, steps_with_probaility),dtype=ms.float32)
    expected_inversion_with_probability[:, 0:prediction_steps] = expected_inversion[:, :]
    some_output = ops.ones((batch_len, mtp_loss.get_output_size()),dtype=ms.float32)
    for i in range(modes):
        # the prediction parts
        some_output[:, steps_with_probaility*i:steps_with_probaility*i+prediction_steps] = \
            expected_inversion[:, :] * i / 5.
        # the probability parts
        some_output[:, steps_with_probaility*i+prediction_steps] = 1./(i+1)

    # computing the loss
    cur_mtp_losses_batch = mtp_loss(some_output, expected_inversion, my_arange)
    cur_mtp_loss = ops.mean(cur_mtp_losses_batch)
    print(cur_mtp_loss)