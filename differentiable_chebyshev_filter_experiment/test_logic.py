import torch
from model import DifferentiableChebyshevFilterLayer

def test_layer_forward():
    in_channels = 2
    order = 2
    batch_size = 4
    length = 40

    layer = DifferentiableChebyshevFilterLayer(in_channels, order=order)
    x = torch.randn(batch_size, in_channels, length)
    y = layer(x)

    assert y.shape == (batch_size, in_channels, length)
    assert not torch.isnan(y).any()
    print("Forward test passed!")

def test_layer_gradients():
    in_channels = 1
    order = 2
    batch_size = 2
    length = 10

    layer = DifferentiableChebyshevFilterLayer(in_channels, order=order)
    x = torch.randn(batch_size, in_channels, length, requires_grad=True)
    y = layer(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.raw_cutoff.grad is not None
    assert layer.raw_ripple.grad is not None

    assert not torch.isnan(layer.raw_cutoff.grad).any()
    assert not torch.isnan(layer.raw_ripple.grad).any()
    print("Gradient test passed!")

def test_layer_different_params():
    in_channels = 1
    x = torch.ones(1, 1, 100)

    layer_low = DifferentiableChebyshevFilterLayer(in_channels, initial_cutoff=0.1)
    layer_high = DifferentiableChebyshevFilterLayer(in_channels, initial_cutoff=0.9)

    with torch.no_grad():
        y_low = layer_low(x)
        y_high = layer_high(x)

    assert not torch.allclose(y_low, y_high)
    print("Parameter sensitivity test passed!")

if __name__ == "__main__":
    test_layer_forward()
    test_layer_gradients()
    test_layer_different_params()
