import torch
import pytest
from model import DifferentiableCopulaLayer

def test_copula_differentiability():
    batch_size = 16
    num_features = 10
    x = torch.randn(batch_size, num_features, requires_grad=True)
    layer = DifferentiableCopulaLayer(num_features, output_type='uniform')

    u = layer(x)
    assert u.shape == (batch_size, num_features)

    loss = u.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0).any()

def test_copula_output_range_uniform():
    batch_size = 100
    num_features = 5
    x = torch.randn(batch_size, num_features)
    layer = DifferentiableCopulaLayer(num_features, output_type='uniform')

    u = layer(x)
    assert (u >= 0).all() and (u <= 1).all()

def test_copula_output_gaussian():
    batch_size = 100
    num_features = 5
    x = torch.randn(batch_size, num_features)
    layer = DifferentiableCopulaLayer(num_features, output_type='gaussian')

    z = layer(x)
    assert z.shape == (batch_size, num_features)
    # Check that it's not all zeros or something weird
    assert torch.abs(z.mean()) < 1.0
    assert z.std() > 0.1

def test_copula_learnable_alpha():
    batch_size = 16
    num_features = 10
    x = torch.randn(batch_size, num_features)
    layer = DifferentiableCopulaLayer(num_features, initial_alpha=10.0, learnable_alpha=True)

    initial_alpha = layer.alpha.clone()
    u = layer(x)
    loss = u.sum()
    loss.backward()

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    optimizer.step()

    assert not torch.equal(layer.alpha, initial_alpha)

if __name__ == "__main__":
    pytest.main([__file__])
