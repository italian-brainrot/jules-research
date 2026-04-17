import torch
from model import DifferentiableGAF

def test_gaf_logic():
    batch_size = 2
    seq_len = 10
    x = torch.linspace(-1, 1, seq_len).repeat(batch_size, 1)

    gaf_layer = DifferentiableGAF(method='summation')
    gaf = gaf_layer(x)

    assert gaf.shape == (batch_size, 1, seq_len, seq_len)

    # Check if GASF is symmetric
    assert torch.allclose(gaf, gaf.transpose(2, 3), atol=1e-6)

    print("GAF Logic Test Passed!")

def test_gaf_differentiability():
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, requires_grad=True)

    gaf_layer = DifferentiableGAF(method='summation')
    gaf = gaf_layer(x)

    loss = gaf.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("GAF Differentiability Test Passed!")

def test_gaf_difference_logic():
    batch_size = 2
    seq_len = 10
    x = torch.linspace(-1, 1, seq_len).repeat(batch_size, 1)

    gaf_layer = DifferentiableGAF(method='difference')
    gaf = gaf_layer(x)

    assert gaf.shape == (batch_size, 1, seq_len, seq_len)

    # GADF: sin(phi_i - phi_j) should be anti-symmetric
    assert torch.allclose(gaf, -gaf.transpose(2, 3), atol=1e-6)

    print("GAF Difference Logic Test Passed!")

if __name__ == "__main__":
    test_gaf_logic()
    test_gaf_differentiability()
    test_gaf_difference_logic()
