import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindvision.classification.models import resnet18, resnet34, resnet50, resnet101, resnet152


def _chop_model(model, remove=1):
    """
    Removes the last layer(s) from the model.
    """
    
    children = list(model.cells())
    model = nn.SequentialCell(children[:-remove])
    return model


def _J(dim):
    res = np.zeros((dim, dim), dtype=np.float32)
    res[:dim // 2, dim // 2:] = np.eye(dim // 2)
    res[dim // 2:, :dim // 2] = -np.eye(dim // 2)
    return res


class SkewSimilarity(nn.Cell):
    def __init__(self, embedding_dim):
        super(SkewSimilarity, self).__init__()
        std = np.sqrt(2.0 / embedding_dim)
        self.J = Parameter(Tensor(np.random.randn(embedding_dim, embedding_dim) * std, dtype=mindspore.float32), name="J")
        self.transpose = ops.Transpose()
        self.matmul = ops.MatMul()
        self.reducesum = ops.ReduceSum(keep_dims=True)

    def construct(self, x, y):
        J_ = 0.5 * (self.J - self.transpose(self.J, (1, 0)))
        result = self.reducesum(self.matmul(x, J_) * y, -1)
        return result


class L2Normalize(nn.Cell):
    def __init__(self, eps=1e-5):
        super(L2Normalize, self).__init__()
        self.eps = eps
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.sqrt = ops.Sqrt()

    def construct(self, x):
        norm = self.sqrt(self.reduce_sum(x * x, -1)) + self.eps
        return x / norm


class ScoreModel(nn.Cell):
    def __init__(self, backbone_model, embedding_dim=512):
        super(ScoreModel, self).__init__()
        self.features = backbone_model
        self.embedding_dim = embedding_dim
        self.norm = L2Normalize()
        self.similarity = SkewSimilarity(embedding_dim=embedding_dim)
        self.reshape = ops.Reshape()

    def construct(self, x):
        B, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]

        inp = self.reshape(x, (-1, C, H, W))

        f = self.features(inp)
        f = self.reshape(f, (B * 2, -1))

        f = self.reshape(f, (B, 2, -1))
        f1, f2 = f[:, 0, :], f[:, 1, :]

        f1 = self.norm(f1)
        f2 = self.norm(f2)

        d = self.similarity(f1, f2)
        return d
    

class Flatten(nn.Cell):
    def __init__(self):
        super(Flatten, self).__init__()
        self.reshape = ops.Reshape()

    def construct(self, x):
        B = x.shape[0]  # 获取批次大小
        return self.reshape(x, (B, -1))  # 展平非批次维度


def get_score_model(name, pretrained=True, path=None):
    if path is None:
        if name == 'resnet18':
            model = resnet18(pretrained=pretrained)
        elif name == 'resnet34':
            model = resnet34(pretrained=pretrained)
        elif name == 'resnet50':
            model = resnet50(pretrained=pretrained)
        elif name == 'resnet101':
            model = resnet101(pretrained=pretrained)
        elif name == 'resnet152':
            model = resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Model {name} is not supported")
    else:
        if name == 'resnet18':
            model = resnet18(pretrained=False)
        elif name == 'resnet34':
            model = resnet34(pretrained=False)
        elif name == 'resnet50':
            model = resnet50(pretrained=False)
        elif name == 'resnet101':
            model = resnet101(pretrained=False)
        elif name == 'resnet152':
            model = resnet152(pretrained=False)
        else:
            raise ValueError(f"Model {name} is not supported")
        
        checkpoint = load_checkpoint(path + f'/{name}.ckpt')
        load_param_into_net(model, checkpoint)

    if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        out_features = model.head.dense.in_channels
        model = _chop_model(model)
        score_model = ScoreModel(model, embedding_dim=out_features)

    return score_model


def main():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    score_model = get_score_model("resnet50", pretrained=True)
    score_model.set_train(False)

    batch_size = 4
    channels, height, width = 3, 224, 224
    input_data = Tensor(np.random.randn(batch_size, 2, channels, height, width).astype(np.float32))

    output = score_model(input_data)

    print("Model output shape:", output.shape)
    print("Model output:", output.asnumpy())


if __name__ == "__main__":
    main()
    