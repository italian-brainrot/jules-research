import torch
import torch.nn as nn
from model import SoftBinningLayer

def test_soft_binning_output_shape():
    num_features = 40
    num_bins = 16
    batch_size = 32
    layer = SoftBinningLayer(num_features, num_bins)
    x = torch.randn(batch_size, num_features)
    output = layer(x)
    assert output.shape == (batch_size, num_features * num_bins)

def test_soft_binning_sum_to_one():
    num_features = 40
    num_bins = 16
    batch_size = 32
    layer = SoftBinningLayer(num_features, num_bins)
    x = torch.randn(batch_size, num_features)
    output = layer(x).view(batch_size, num_features, num_bins)
    # Each feature's soft-bin assignments should sum to 1
    sums = output.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))

def test_soft_binning_gradient_flow():
    num_features = 40
    num_bins = 16
    batch_size = 32
    layer = SoftBinningLayer(num_features, num_bins)
    x = torch.randn(batch_size, num_features, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Check if gradients flow to x
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Check if gradients flow to parameters
    assert layer.centers.grad is not None
    assert layer.centers.grad.abs().sum() > 0
    assert layer.log_temperature.grad is not None
    assert layer.log_temperature.grad.abs().sum() > 0

def test_soft_binning_independence():
    # If we change feature i, it should only affect bin i in output
    num_features = 5
    num_bins = 4
    # Use higher temperature to avoid extreme softmax
    layer = SoftBinningLayer(num_features, num_bins, temperature=1.0)
    x1 = torch.randn(1, num_features)
    x2 = x1.clone()
    x2[0, 2] += 0.5 # Change feature 2

    out1 = layer(x1).view(num_features, num_bins)
    out2 = layer(x2).view(num_features, num_bins)

    for i in range(num_features):
        if i == 2:
            # Should be different
            assert not torch.allclose(out1[i], out2[i], atol=1e-5)
        else:
            # Should be identical
            assert torch.allclose(out1[i], out2[i], atol=1e-7)

if __name__ == "__main__":
    test_soft_binning_output_shape()
    test_soft_binning_sum_to_one()
    test_soft_binning_gradient_flow()
    test_soft_binning_independence()
    print("All tests passed!")
