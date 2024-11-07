import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class CrossTransformer(nn.Cell):
    def __init__(self, dim=512, dim_key=128, dim_value=128):
        super(CrossTransformer, self).__init__()
        self.scale = dim_key ** -0.5
        self.to_qk = nn.Conv2d(dim, dim_key, kernel_size=1, has_bias=False)
        self.to_v = nn.Conv2d(dim, dim_value, kernel_size=1, has_bias=False)

    def construct(self, model, img_query, img_supports):
        b, k, n, c, h, w = img_supports.shape

        query_repr = model(img_query)  
        b, c, h, w = query_repr.shape
        supports_repr = model(img_supports.view(b * k * n, c, h, w))  
        
        query_q = self.to_qk(query_repr)
        query_v = self.to_v(query_repr)

        supports_k = self.to_qk(supports_repr)
        supports_v = self.to_v(supports_repr) 
        
        x = supports_k
        x_split = ops.Split(axis=0, output_num=b)(x)  
        x_split_reshaped = [ops.Split(axis=0, output_num=k)(chunk) for chunk in x_split]
        supports_k = ops.Stack(axis=0)([ops.Stack(axis=0)(inner_split) for inner_split in x_split_reshaped])
        y = supports_v
        y_split = ops.Split(axis=0, output_num=b)(y)  
        y_split_reshaped = [ops.Split(axis=0, output_num=k)(chunk) for chunk in y_split]
        supports_v = ops.Stack(axis=0)([ops.Stack(axis=0)(inner_split) for inner_split in y_split_reshaped])
        
        b, k, n, c, i, j = supports_k.shape
        query_q_reshaped = query_q.reshape(b, c, h * w).transpose(0, 2, 1)  
        supports_k_reshaped = supports_k.reshape(b, k, n, c, i * j).transpose(0, 1, 2, 4, 3)  
        broadcast_to = ops.BroadcastTo(supports_k_reshaped.shape)
        query_q_broadcasted = broadcast_to(query_q_reshaped)
        batch_matmul = ops.BatchMatMul(transpose_b=True)
        
        sim = batch_matmul(query_q_broadcasted, supports_k_reshaped)
        sim = sim.reshape(b, h, w, k, n, i, j).transpose(0, 3, 1, 2, 4, 5, 6)  
        sim = sim.view(b, k, h, w, n*i*j)

        attn = sim.softmax(axis=-1)
        attn_reshaped = ops.reshape(attn, (b, k, h * w, n * i * j))  
       
        supports_v_reshaped = ops.reshape(supports_v, (b, k, n * i * j, c))  
        out = ops.matmul(attn_reshaped, supports_v_reshaped)
        out = ops.reshape(out, (b, k, c, h, w))
        
        out = out.view(b, k, c*h*w)
        query_v_reshaped = ops.reshape(query_v, (b, c * h * w)) 
        query_v = ops.expand_dims(query_v_reshaped, 1)
        
        euclidean_dist = ((query_v - out) ** 2).sum(axis = -1) / (h * w)
        return -euclidean_dist
       
if __name__ == "__main__":
    batch_size, num_classes, num_images = 1, 10, 5
    channels, height, width = 512, 28, 28
    img_query = Tensor(ms.numpy.randn(batch_size, channels, height, width), ms.float32)
    img_supports = Tensor(ms.numpy.randn(batch_size, num_classes, num_images, channels, height, width), ms.float32)
    model = nn.Cell()
    model.construct = lambda x: x  
    transformer = CrossTransformer(dim=512, dim_key=128, dim_value=128)
    dist = transformer(model, img_query, img_supports)
    print(dist.shape)