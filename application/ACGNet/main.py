import mindspore as ms
from mindspore import ops

from model import ACGNet

def test_acgnet():
    b, t, c = 4, 100, 8  # Batch size 4, time steps 100, feature size 8

    ms.set_seed(42)
    x = ops.randn((b, t, c), seed=42)  

    model = ACGNet(num_layers=2, hid_dim=8)

    original_features, updated_features, adjacency_matrix = model(ms.Tensor(x, ms.float32))

    # 验证输出形状
    assert original_features.shape == (b, t, c), \
        f"Expected original features shape {(b, t, c)}, but got {original_features.shape}"
    assert updated_features.shape == (b, t, c), \
        f"Expected updated features shape {(b, t, c)}, but got {updated_features.shape}"
    assert adjacency_matrix.shape == (b, t, t), \
        f"Expected adjacency matrix shape {(b, t, t)}, but got {adjacency_matrix.shape}"


    print("Original Features (x):", original_features.asnumpy())
    print("Updated Features (x + F_avg + F_conv):", updated_features.asnumpy())
    print("Adjacency Matrix (A_prime):", adjacency_matrix.asnumpy())

    assert isinstance(original_features, ms.Tensor), "Original features must be a Tensor"
    assert isinstance(updated_features, ms.Tensor), "Updated features must be a Tensor"
    assert isinstance(adjacency_matrix, ms.Tensor), "Adjacency matrix must be a Tensor"
    assert adjacency_matrix.min() >= 0, "Adjacency matrix should contain non-negative values"

    print("Test passed. All outputs have correct shapes and expected properties.")

if __name__ == "__main__":
    test_acgnet()

