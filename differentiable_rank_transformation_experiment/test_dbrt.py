import torch
import pytest
from model import DifferentiableBatchRank

def test_dbrt_shape():
    batch_size = 16
    num_features = 10
    layer = DifferentiableBatchRank(num_features)
    x = torch.randn(batch_size, num_features)
    output = layer(x)
    assert output.shape == (batch_size, num_features)

def test_dbrt_range():
    batch_size = 32
    num_features = 5
    layer = DifferentiableBatchRank(num_features, initial_alpha=100.0) # High alpha for sharp ranks
    x = torch.randn(batch_size, num_features)
    output = layer(x)
    # Output should be between 0 and 1
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)

def test_dbrt_monotonicity():
    # If we increase one element in the feature, its rank should increase
    batch_size = 10
    num_features = 1
    layer = DifferentiableBatchRank(num_features, initial_alpha=10.0)
    x = torch.linspace(0, 1, batch_size).unsqueeze(1) # [0, 0.1, ..., 0.9]
    output = layer(x)

    # Ranks should be strictly increasing
    assert torch.all(torch.diff(output.squeeze()) > 0)

def test_dbrt_differentiability():
    batch_size = 8
    num_features = 4
    layer = DifferentiableBatchRank(num_features)
    x = torch.randn(batch_size, num_features, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.alpha.grad is not None

if __name__ == "__main__":
    # Run tests manually
    test_dbrt_shape()
    test_dbrt_range()
    test_dbrt_monotonicity()
    test_dbrt_differentiability()
    print("All DBRT tests passed!")
