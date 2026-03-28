import torch
from differentiable_gabor_filter_experiment.model import GaborLayer, GaborConvMLP

def test_gabor_layer_differentiability():
    print("Testing GaborLayer differentiability...")
    in_channels = 1
    out_channels = 8
    kernel_size = 15
    batch_size = 4
    signal_length = 40

    layer = GaborLayer(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    x = torch.randn(batch_size, in_channels, signal_length, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.frequencies.grad is not None
    assert layer.sigmas.grad is not None
    assert layer.phases.grad is not None

    print("GaborLayer differentiability test passed.")

def test_gabor_conv_mlp_shapes():
    print("Testing GaborConvMLP shapes...")
    input_dim = 40
    hidden_dim = 64
    output_dim = 10
    batch_size = 4

    model = GaborConvMLP(input_dim, hidden_dim, output_dim)
    x = torch.randn(batch_size, input_dim)

    output = model(x)

    assert output.shape == (batch_size, output_dim)
    print("GaborConvMLP shapes test passed.")

if __name__ == "__main__":
    test_gabor_layer_differentiability()
    test_gabor_conv_mlp_shapes()
