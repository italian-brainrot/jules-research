import torch
import numpy as np
from model import FrFTLayer

def test_frft_identity():
    n = 10
    layer = FrFTLayer(n, num_orders=1)
    # Set alpha to 0
    with torch.no_grad():
        layer.alphas.fill_(0.0)

    x = torch.randn(5, n)
    out = layer(x) # (batch, 1 * n)
    out = out.reshape(5, n)

    # At alpha=0, FrFT is identity.
    # Our layer returns magnitude, so it should be abs(x).
    # Since x is real, abs(x) might not be x.
    # Actually, FrFT of a real signal at alpha=0 is the signal itself.
    # Our layer computes torch.abs(FrFT(x)).

    assert torch.allclose(out, torch.abs(x), atol=1e-5)

def test_frft_dft():
    n = 16
    layer = FrFTLayer(n, num_orders=1)
    # Set alpha to 1
    with torch.no_grad():
        layer.alphas.fill_(1.0)

    x = torch.randn(5, n)
    out = layer(x).reshape(5, n)

    # At alpha=1, FrFT is DFT.
    # Standard DFT in torch: torch.fft.fft
    expected = torch.abs(torch.fft.fft(x, norm='ortho'))

    # Note: DFT matrix we used was (1/sqrt(n)) * exp(-2pi * i * m * n / n)
    # torch.fft.fft with norm='ortho' should match this.

    assert torch.allclose(out, expected, atol=1e-5)

def test_frft_differentiability():
    n = 8
    layer = FrFTLayer(n, num_orders=2)
    x = torch.randn(2, n, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.alphas.grad is not None
    assert not torch.isnan(layer.alphas.grad).any()
    assert not torch.isnan(x.grad).any()

if __name__ == "__main__":
    test_frft_identity()
    test_frft_dft()
    test_frft_differentiability()
    print("All tests passed!")
