import torch
from model import DZTLayer, DZTAugmentedMLP

def test_dzt_layer_shapes():
    batch_size = 16
    input_len = 40
    num_points = 20
    layer = DZTLayer(input_len, num_points)
    x = torch.randn(batch_size, input_len)
    out = layer(x)
    assert out.shape == (batch_size, 2 * num_points)
    print("Shape test passed.")

def test_dzt_layer_gradients():
    batch_size = 4
    input_len = 10
    num_points = 5
    layer = DZTLayer(input_len, num_points)
    x = torch.randn(batch_size, input_len, requires_grad=True)

    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert layer.gamma.grad is not None
    assert layer.omega.grad is not None
    print("Gradient test passed.")

def test_dzt_augmented_mlp_shapes():
    batch_size = 16
    input_len = 40
    num_points = 20
    hidden_dim = 64
    output_dim = 10
    model = DZTAugmentedMLP(input_len, num_points, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_len)
    out = model(x)
    assert out.shape == (batch_size, output_dim)
    print("Augmented MLP shape test passed.")

if __name__ == "__main__":
    test_dzt_layer_shapes()
    test_dzt_layer_gradients()
    test_dzt_augmented_mlp_shapes()
