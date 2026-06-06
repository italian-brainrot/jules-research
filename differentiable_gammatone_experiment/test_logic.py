import torch
import pytest
from differentiable_gammatone_experiment.model import GammatoneLayer

def test_gammatone_differentiability():
    batch_size = 2
    in_channels = 1
    out_channels = 4
    seq_len = 40
    kernel_size = 15

    layer = GammatoneLayer(in_channels, out_channels, kernel_size)
    x = torch.randn(batch_size, in_channels, seq_len, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.frequencies.grad is not None
    assert layer.bandwidths.grad is not None
    assert layer.phases.grad is not None

    # Check if gradients are non-zero
    assert torch.sum(torch.abs(layer.frequencies.grad)) > 0
    assert torch.sum(torch.abs(layer.bandwidths.grad)) > 0
    assert torch.sum(torch.abs(layer.phases.grad)) > 0

def test_gammatone_shapes():
    batch_size = 2
    in_channels = 1
    out_channels = 8
    seq_len = 40
    kernel_size = 15

    layer = GammatoneLayer(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    x = torch.randn(batch_size, in_channels, seq_len)

    output = layer(x)
    assert output.shape == (batch_size, out_channels, seq_len)

def test_gammatone_kernel_normalization():
    layer = GammatoneLayer(1, 4, 15)
    kernels = layer.get_kernels()

    # Check L2 norm
    norms = torch.norm(kernels, p=2, dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__])
