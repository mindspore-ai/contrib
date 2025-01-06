import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal

class LangModelWithDense(nn.Cell):
    def __init__(self, lang_model, input_size, hidden_size, fine_tune):
        super(LangModelWithDense, self).__init__()
        self.lang_model = lang_model

        self.linear1 = nn.Dense(input_size, hidden_size, weight_init=TruncatedNormal(0.02))
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Dense(hidden_size, 1, weight_init=TruncatedNormal(0.02))

        self.fine_tune = fine_tune

    def construct(self, x, mask):
        if self.fine_tune:
            embeddings = self.lang_model(x, attention_mask=mask)[0][:, 0, :]

        else:
            self.lang_model.set_train(False)
            embeddings = self.lang_model(x, attention_mask=mask)[0][:, 0, :]

        output = self.dropout1(ops.GeLU()(self.linear1(embeddings)))

        output = self.linear2(output)

        return output