import torch
import pytest
from model import TeagerKaiserLayer

def test_tkeo_logic():
    # Simple test case: x = [1, 2, 4, 7]
    # psi(2) = 2^2 - 1*4 = 0
    # psi(4) = 4^2 - 2*7 = 16 - 14 = 2
    # With replicate padding: [1, 1, 2, 4, 7, 7]
    # psi(1) = 1^2 - 1*2 = -1
    # psi(2) = 2^2 - 1*4 = 0
    # psi(4) = 4^2 - 2*7 = 2
    # psi(7) = 7^2 - 4*7 = 49 - 28 = 21

    layer = TeagerKaiserLayer(in_channels=1, smooth_kernel_size=1) # No smoothing for pure TKEO test
    x = torch.tensor([[[1.0, 2.0, 4.0, 7.0]]])
    expected = torch.tensor([[[-1.0, 0.0, 2.0, 21.0]]])

    out = layer(x)
    assert torch.allclose(out, expected)

def test_tkeo_differentiability():
    layer = TeagerKaiserLayer(in_channels=1, smooth_kernel_size=3)
    x = torch.randn(2, 1, 40, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Check if smoothing weights get gradients
    for param in layer.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

def test_tkeo_zero_signal():
    layer = TeagerKaiserLayer(in_channels=1)
    x = torch.zeros(1, 1, 10)
    out = layer(x)
    assert torch.all(out == 0)

if __name__ == "__main__":
    test_tkeo_logic()
    test_tkeo_differentiability()
    test_tkeo_zero_signal()
    print("All tests passed!")
