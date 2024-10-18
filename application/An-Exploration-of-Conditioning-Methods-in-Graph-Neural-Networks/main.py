import mindspore
from mindspore import nn
import mindspore.ops as ops

# from layers import *
from layers import ConditionalLinear
import numpy as np

def main():
    in_features = 10
    out_features = 5
    cond_features = 3

    conditional_layer = ConditionalLinear(in_features, out_features, cond_features, method='pure')

    f_in = mindspore.Tensor(np.random.randn(2, in_features).astype(np.float32))  # 批大小为 2，特征数量为 in_features
    cond_vec = mindspore.Tensor(np.random.randn(2, cond_features).astype(np.float32))  # 同样的批大小，特征数量为 cond_features

    output = conditional_layer(f_in, cond_vec)
    
    print("输出:")
    print(output)

if __name__ == "__main__":
    main()
    