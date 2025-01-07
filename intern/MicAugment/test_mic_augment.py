import numpy as np
import mindspore as ms
from mic_augment_mindspore import MicrophoneModel


if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE)

    hparams = {
        'win_length': 1024,
        'n_fft': 1024,
        'hop_length': 256,
        'lr': 1e-4,
        'beta1': 0.5,
        'beta2': 0.9,
    }

    audio = ms.Tensor(np.random.randn(3, 16384).astype(np.float32))
    microphone_model = MicrophoneModel(hparams)
    audio_mic = microphone_model(audio)

    print(f'audio out size: {audio_mic.shape}')
    print('DONE')