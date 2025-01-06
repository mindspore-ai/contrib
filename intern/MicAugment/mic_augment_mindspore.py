import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal
import librosa


def smoothmax(a, b):
    exp = ops.Exp()
    exp_a = exp(a)
    exp_b = exp(b)
    numerator = a * exp_a + b * exp_b
    denominator = exp_a + exp_b
    return numerator / denominator


def smoothmin(a, b):
    exp = ops.Exp()
    exp_m_a = exp(-a)
    exp_m_b = exp(-b)
    numerator = a * exp_m_a + b * exp_m_b
    denominator = exp_m_a + exp_m_b
    return numerator / denominator


class AudioProcessor(nn.Cell):
    def __init__(self, n_fft, hop_length, win_length):
        super(AudioProcessor, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def stft(self, audio):
        audio_np = audio.asnumpy()
        window = np.hanning(self.win_length)
        stfts = []
        for i in range(audio_np.shape[0]):
            stft = librosa.stft(
                audio_np[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window
            )
            stfts.append(stft)
        stft_data = np.stack(stfts)
        return ms.Tensor(stft_data, dtype=ms.float32)

    def istft(self, stft_matrix):
        stft_np = stft_matrix.asnumpy()
        window = np.hanning(self.win_length)
        waves = []
        for i in range(stft_np.shape[0]):
            wave = librosa.istft(
                stft_np[i],
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window
            )
            waves.append(wave)
        wave_data = np.stack(waves)
        return ms.Tensor(wave_data, dtype=ms.float32)


class MicrophoneModel(nn.Cell):
    def __init__(self, hparams):
        super(MicrophoneModel, self).__init__()
        self.win_length = hparams['win_length']
        self.hop_length = hparams['hop_length']
        self.n_fft = hparams['n_fft']
        self.lr = hparams['lr']
        self.beta1 = hparams['beta1']
        self.beta2 = hparams['beta2']

        stft_dim = int(self.n_fft / 2) + 1
        self.impulse_response = ms.Parameter(
            ms.Tensor(np.random.randn(stft_dim, 1).astype(np.float32))
        )
        self.threshold = ms.Parameter(
            ms.Tensor(np.random.randn(stft_dim, 1).astype(np.float32))
        )
        self.filter = ms.Parameter(
            ms.Tensor(np.random.randn(stft_dim, 1).astype(np.float32))
        )
        self.mic_clip = ms.Parameter(
            ms.Tensor(np.random.randn(1).astype(np.float32))
        )

        self.audio_processor = AudioProcessor(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        self.sigmoid = ops.Sigmoid()
        self.abs = ops.Abs()

    def construct(self, x):
        x_cmplx = self.audio_processor.stft(x)
        y_1 = self.impulse_response * x_cmplx
        abs_y1_squared = self.abs(y_1) ** 2
        threshold_expanded = ops.broadcast_to(self.threshold, y_1.shape)
        y_2 = self.audio_processor.istft(
            y_1 * self.sigmoid(abs_y1_squared - threshold_expanded)
        )
        noise = ops.StandardNormal()(y_2.shape)
        noise_stft = self.audio_processor.stft(noise)
        filtered_noise = self.audio_processor.istft(noise_stft * self.filter)
        y_3 = y_2 + filtered_noise
        y = smoothmin(smoothmax(y_3, -self.mic_clip), self.mic_clip)
        return y

    def get_optimizer(self):
        return nn.Adam(
            self.trainable_params(),
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2
        )