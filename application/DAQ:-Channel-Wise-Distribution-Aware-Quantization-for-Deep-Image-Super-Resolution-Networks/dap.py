import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

class RoundDiff(nn.Cell):
    def __init__(self):
        super(RoundDiff, self).__init__()
        self.round = ops.Round()

    def construct(self, x):
        return self.round(x)

    def bprop(self, x, out, dout):
        return (dout,)

class DAQBase:
    def __init__(self, n_bits=8):
        self.step_sizes = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.335, 8: 0.031}
        self.n_bits = n_bits
        self.step_size = self.step_sizes[n_bits]
        self.alpha = (2 ** (self.n_bits - 1) - 0.5) * self.step_size
        self.beta = 0

        self.mean = 0
        self.sigma = 0

    def quantize(self, x: ms.Tensor):
        pass

class DAQActivations(DAQBase):
    def __init__(self, n_bits=8, daq_input=False):
        super().__init__(n_bits=n_bits)
        self.daq_input = daq_input
        self.round_diff = RoundDiff()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.maximum = ops.Maximum()
        self.zeros_like = ops.ZerosLike()
        self.sqrt = ops.Sqrt()

    def compute_std(self, x: ms.Tensor, dims) -> ms.Tensor:
        mean = self.reduce_mean(x, dims)
        squared_diff = ops.square(x - mean)
        variance = self.reduce_mean(squared_diff, dims)
        return self.sqrt(variance)

    def adaptive_transformer(self, x: ms.Tensor) -> ms.Tensor:
        self.mean = self.reduce_mean(x, (0, 2, 3))
        self.sigma = self.compute_std(x, (0, 2, 3))
        return (x - self.mean) / (self.sigma + 1e-7)

    def inv_adaptive_transformer(self, x_q_hat: ms.Tensor) -> ms.Tensor:
        return x_q_hat * self.sigma + self.mean

    def clamp(self, x: ms.Tensor) -> ms.Tensor:
        self.update_beta()
        x_clamped = self.zeros_like(x)
        
        for c in range(x.shape[1]):
            x_slice = x[:, c:c+1, :, :]
            beta_value = self.beta[0, c:c+1, 0, 0]
            x_clamped[:, c:c+1, :, :] = ops.clip_by_value(
                x_slice,
                ms.Tensor(-self.alpha + beta_value, dtype=ms.float32),
                ms.Tensor(self.alpha + beta_value, dtype=ms.float32)
            )
        return x_clamped

    def update_beta(self):
        if self.daq_input:
            self.beta = self.zeros_like(self.sigma)
        else:
            self.beta = self.maximum(
                self.alpha - self.mean / (self.sigma + 1e-7),
                self.zeros_like(self.sigma)
            )

    def adaptive_discretizer(self, x_hat: ms.Tensor) -> ms.Tensor:
        return (self.round_diff((self.clamp(x_hat) / self.step_size) + 0.5) - 0.5) * self.step_size

    def quantize(self, x: ms.Tensor) -> ms.Tensor:
        x_hat = self.adaptive_transformer(x)
        x_q_hat = self.adaptive_discretizer(x_hat)
        x_q = self.inv_adaptive_transformer(x_q_hat)
        return x_q

class DAQWeights(DAQBase):
    def __init__(self, n_bits=8):
        super().__init__(n_bits=n_bits)
        self.round_diff = RoundDiff()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.sqrt = ops.Sqrt()

    def compute_std(self, x: ms.Tensor, dim) -> ms.Tensor:
        mean = self.reduce_mean(x, dim)
        squared_diff = ops.square(x - mean)
        variance = self.reduce_mean(squared_diff, dim)
        return self.sqrt(variance)

    def adaptive_transformer(self, x: ms.Tensor) -> ms.Tensor:
        # weight quantizer is not adaptive to channels
        self.sigma = self.compute_std(x, 0)
        return x / (self.sigma + 1e-7)

    def inv_adaptive_transformer(self, x_q_hat: ms.Tensor) -> ms.Tensor:
        return x_q_hat * self.sigma

    def clamp(self, x: ms.Tensor) -> ms.Tensor:
        return ops.clip_by_value(
            x,
            ms.Tensor(-self.alpha, dtype=ms.float32),
            ms.Tensor(self.alpha, dtype=ms.float32)
        )

    def adaptive_discretizer(self, x_hat: ms.Tensor) -> ms.Tensor:
        return (self.round_diff((self.clamp(x_hat) / self.step_size) + 0.5) - 0.5) * self.step_size

    def quantize(self, x: ms.Tensor) -> ms.Tensor:
        x_hat = self.adaptive_transformer(x)
        x_q_hat = self.adaptive_discretizer(x_hat)
        x_q = self.inv_adaptive_transformer(x_q_hat)
        return x_q

def main():
    print("测试权重量化 (DAQWeights):")
    print("-" * 50)
    
    # 测试不同大小的权重
    weight_shapes = [
        (32, 16, 3, 3),    # 标准卷积层权重
        (64, 32, 5, 5),    # 较大卷积核
        (128, 64, 1, 1),   # 1x1卷积
        (256, 128, 7, 7),  # 大卷积核
        (512, 256, 3, 3)   # 深层卷积
    ]
    
    for bits in [2, 4, 8]:
        print(f"\n使用 {bits} 位量化:")
        daq_w = DAQWeights(n_bits=bits)
        
        for shape in weight_shapes:
            w = ms.Tensor(np.random.randn(*shape), dtype=ms.float32)
            w_q = daq_w.quantize(w)
            mse = np.mean((w.asnumpy() - w_q.asnumpy()) ** 2)
            print(f"权重形状 {shape}:")
            print(f"  - 均方误差: {mse:.6f}")
            print(f"  - 原始范围: [{w.asnumpy().min():.3f}, {w.asnumpy().max():.3f}]")
            print(f"  - 量化范围: [{w_q.asnumpy().min():.3f}, {w_q.asnumpy().max():.3f}]")
    
    print("\n测试激活量化 (DAQActivations):")
    print("-" * 50)
    
    # 测试不同大小的特征图
    feature_shapes = [
        (1, 3, 32, 32),     # 小型输入
        (1, 16, 64, 64),    # 中型特征图
        (1, 32, 128, 128),  # 大型特征图
        (1, 64, 56, 56),    # ResNet风格
        (1, 256, 28, 28)    # 深层特征图
    ]
    
    for bits in [2, 4, 8]:
        print(f"\n使用 {bits} 位量化:")
        
        # 测试普通激活
        print("\n普通激活量化:")
        daq_act = DAQActivations(n_bits=bits, daq_input=False)
        
        for shape in feature_shapes:
            x = ms.Tensor(np.random.randn(*shape), dtype=ms.float32)
            x_q = daq_act.quantize(x)
            mse = np.mean((x.asnumpy() - x_q.asnumpy()) ** 2)
            print(f"特征图形状 {shape}:")
            print(f"  - 均方误差: {mse:.6f}")
            print(f"  - 原始范围: [{x.asnumpy().min():.3f}, {x.asnumpy().max():.3f}]")
            print(f"  - 量化范围: [{x_q.asnumpy().min():.3f}, {x_q.asnumpy().max():.3f}]")
        
        # 测试输入量化
        print("\n输入量化:")
        daq_input = DAQActivations(n_bits=bits, daq_input=True)
        
        for shape in feature_shapes[:2]:  # 只测试较小的输入形状
            x = ms.Tensor(np.random.randn(*shape), dtype=ms.float32)
            x_q = daq_input.quantize(x)
            mse = np.mean((x.asnumpy() - x_q.asnumpy()) ** 2)
            print(f"输入形状 {shape}:")
            print(f"  - 均方误差: {mse:.6f}")
            print(f"  - 原始范围: [{x.asnumpy().min():.3f}, {x.asnumpy().max():.3f}]")
            print(f"  - 量化范围: [{x_q.asnumpy().min():.3f}, {x_q.asnumpy().max():.3f}]")

if __name__ == "__main__":
    main() 
