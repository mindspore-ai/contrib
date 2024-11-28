import json
import mindspore as ms
import mindspore.nn as nn
from mindnlp.transformers import Wav2Vec2Model, Wav2Vec2Config


class AvesClassifier(nn.Cell):
    def __init__(self, config_path, model_path=None):
        super(AvesClassifier, self).__init__()

        self.config = self.load_config(config_path)
        wav2vec2_config = Wav2Vec2Config(**self.config)

        self.model = Wav2Vec2Model(wav2vec2_config)
        if model_path:
            self.load_model(model_path)

        for param in self.model.feature_extractor.trainable_params():
            param.requires_grad = False
            

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def load_model(self, model_path):
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(self.model, param_dict)

    def construct(self, sig):
        out = self.model(sig).last_hidden_state
        return out


if __name__ == '__main__':
    from mindspore import ops
    
    config_path = 'aves-base-bio.json'
    model = AvesClassifier(config_path)
    model.set_train(False)

    waveform = ops.rand((16_000))
    x = waveform.expand_dims(0)
    out = model(x)
    print(out.shape)
    print(out)
