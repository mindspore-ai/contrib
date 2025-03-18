import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
import mindspore.numpy as mnp


def entropy_loss(x):
    x1 = x.reshape((x.shape[0], -1))
    eps = float(np.finfo(x.asnumpy().dtype).eps)
    min_val = float(np.finfo(x.asnumpy().dtype).min)
    max_val = float(np.finfo(x.asnumpy().dtype).max)
    probs = x1 / (x1.sum(axis=-1, keepdims=True) + Tensor(eps, x.dtype))
    log_probs = ops.Log()(probs + Tensor(eps, x.dtype))
    clamped_log = ops.clip_by_value(log_probs, Tensor(min_val, x.dtype))
    entropy = -(probs * clamped_log).sum(axis=-1)
    return entropy.mean()


def cyclostationary(x):
    x = x ** 2
    eps = float(np.finfo(x.asnumpy().dtype).eps)
    std_val = mnp.std(x, axis=-1)
    mean_val = mnp.mean(x, axis=-1)
    x = std_val / (mean_val + eps)
    fft_result = mnp.fft.rfft(x, norm="forward")
    x = mnp.abs(mnp.mean(fft_result))
    return x


def kurtosis_loss(x):
    kur = mnp.mean(x ** 4, axis=1) / (mnp.mean(x ** 2, axis=1) ** 2)
    return mnp.mean(kur)


def intelligent_spectrogram_loss(x):
    x = mnp.abs(x)
    eps = float(np.finfo(x.asnumpy().dtype).eps)
    c_m = mnp.std(x, axis=-2) / (mnp.mean(x, axis=-2) + eps)
    c_k = mnp.std(x, axis=-1) / (mnp.mean(x, axis=-1) + eps)
    q_f = mnp.mean(c_m / (mnp.max(c_m) + eps))
    q_t = mnp.mean(c_k / (mnp.max(c_k) + eps))
    q_ft = q_f * q_t
    loss = mnp.exp((q_f + q_t + q_ft + mnp.abs(q_f - q_t) + mnp.abs(q_f - q_ft) + mnp.abs(q_t - q_ft)) / 6.0) - 1
    return loss


class Intelligent_spectrogram_loss_1(nn.Cell):
    def __init__(self, a=1):
        super(Intelligent_spectrogram_loss, self).__init__()
        self.a = a

    def construct(self, x):
        x = mnp.abs(x)
        eps = float(np.finfo(x.asnumpy().dtype).eps)
        c_m = mnp.mean(x, axis=-2) / (mnp.std(x, axis=-2) + eps)
        c_k = mnp.mean(x, axis=-1) / (mnp.std(x, axis=-1) + eps)
        q_f = mnp.mean(c_m / (mnp.max(c_m) + eps))
        q_t = mnp.mean(c_k / (mnp.max(c_k) + eps))
        q_ft = q_f * q_t
        return 1 / q_ft


class Intelligent_spectrogram_loss(nn.Cell):
    def __init__(self):
        super(Intelligent_spectrogram_loss, self).__init__()
    
    def construct(self, x):
        x = mnp.abs(x)
        eps = float(np.finfo(x.asnumpy().dtype).eps)
        c_m = mnp.mean(x, axis=-2) / (mnp.std(x, axis=-2) + eps)
        c_k = mnp.mean(x, axis=-1) / (mnp.std(x, axis=-1) + eps)
        q_f = mnp.mean(c_m / (mnp.max(c_m) + eps))
        q_t = mnp.mean(c_k / (mnp.max(c_k) + eps))
        return 2.0 * q_f * q_t / (q_f + q_t)


class Calculate_renyi_entropy(nn.Cell):
    def __init__(self, bins=50, alpha=2):
        super(Calculate_renyi_entropy, self).__init__()
        self.bins = bins
        self.alpha = alpha

    def construct(self, time_freq_map):
        time_freq_map = time_freq_map[0]
        time_freq_np = time_freq_map.asnumpy()
        hist, _ = np.histogram(time_freq_np, bins=self.bins)
        histogram = mindspore.Tensor(hist, dtype=time_freq_map.dtype)
        probability = histogram / mnp.sum(histogram)
        renyi_entropy = (1 / (1 - self.alpha)) * mnp.log2(mnp.sum(probability ** self.alpha))
        return renyi_entropy
