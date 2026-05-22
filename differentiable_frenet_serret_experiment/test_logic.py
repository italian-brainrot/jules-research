import torch
import pytest
from model import DifferentiableFrenetSerret

def test_fs_layer_shapes():
    batch_size = 8
    seq_len = 40
    layer = DifferentiableFrenetSerret()
    x = torch.randn(batch_size, seq_len)
    features = layer(x)

    # We defined 6 features: mean/max/std for kappa and torsion
    assert features.shape == (batch_size, 6)

def test_fs_layer_differentiability():
    batch_size = 4
    seq_len = 40
    layer = DifferentiableFrenetSerret()
    x = torch.randn(batch_size, seq_len, requires_grad=True)

    features = layer(x)
    loss = features.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.sigma.grad is not None
    assert layer.tau_delay.grad is not None

    print(f"Sigma grad: {layer.sigma.grad.item()}")
    print(f"Tau grad: {layer.tau_delay.grad.item()}")

def test_fs_layer_invariance():
    batch_size = 2
    seq_len = 40
    layer = DifferentiableFrenetSerret()
    x = torch.randn(batch_size, seq_len)

    features1 = layer(x)
    features2 = layer(x + 5.0) # Constant offset

    # Should be almost identical (up to numerical precision)
    print(f"Features 1: {features1}")
    print(f"Features 2: {features2}")
    print(f"Diff: {torch.abs(features1 - features2).max().item()}")
    assert torch.allclose(features1, features2, atol=1e-3)

if __name__ == "__main__":
    test_fs_layer_shapes()
    test_fs_layer_differentiability()
    test_fs_layer_invariance()
    print("All tests passed!")
