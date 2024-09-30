import mindspore as ms
from mindspore import ops,nn

class EfficientAttention(nn.Cell):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def construct(self, input_):
        n, _, h, w = input_.shape
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = ops.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], axis=2)
            query = ops.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], axis=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.swapaxes(1, 2)
            attended_value = (
                context.swapaxes(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = ops.cat(attended_values, axis=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention
    
if __name__ == '__main__':
    x = ms.tensor([[[[1, 1], [1, 1]]]], dtype=ms.float32)
    print(EfficientAttention(1, 2, 2, 2)(x))