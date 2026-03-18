import torch
from model import DifferentiableShapeletLayer

def test_dsl_shape():
    batch_size = 4
    in_channels = 1
    signal_length = 40
    num_shapelets = 8
    shapelet_length = 10

    x = torch.randn(batch_size, in_channels, signal_length)
    layer = DifferentiableShapeletLayer(in_channels, num_shapelets, shapelet_length)

    output = layer(x)
    assert output.shape == (batch_size, num_shapelets)
    print("Shape test passed!")

def test_dsl_gradients():
    batch_size = 2
    in_channels = 1
    signal_length = 20
    num_shapelets = 4
    shapelet_length = 5

    x = torch.randn(batch_size, in_channels, signal_length, requires_grad=True)
    layer = DifferentiableShapeletLayer(in_channels, num_shapelets, shapelet_length)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.shapelets.grad is not None
    print("Gradients test passed!")

def test_dsl_distance():
    # Test if distance calculation is roughly correct
    in_channels = 1
    num_shapelets = 1
    shapelet_length = 5

    layer = DifferentiableShapeletLayer(in_channels, num_shapelets, shapelet_length)
    # Set shapelet to zeros
    layer.shapelets.data.zero_()

    # Input with one window being zeros
    x = torch.ones(1, 1, 10)
    x[0, 0, 2:7] = 0.0 # This window matches the shapelet perfectly

    output = layer(x)
    # Since one window is perfectly matching (dist=0), the soft-min should be low
    print(f"Distance to zero shapelet with matching window: {output.item()}")
    assert output.item() < 1.0 # Should be small, but temperature matters

if __name__ == "__main__":
    test_dsl_shape()
    test_dsl_gradients()
    test_dsl_distance()
