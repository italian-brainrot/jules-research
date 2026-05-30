import torch
from model import DGLCMLayer

def test_glcm_logic():
    batch_size = 4
    seq_len = 40
    num_bins = 8
    offsets = [1, 2, 5]

    layer = DGLCMLayer(num_bins=num_bins, offsets=offsets)
    x = torch.randn(batch_size, seq_len, requires_grad=True)

    out = layer(x)

    # Check output shape: (batch_size, len(offsets) * 5)
    expected_shape = (batch_size, len(offsets) * 5)
    assert out.shape == expected_shape

    # Check gradient flow to input
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Check gradient flow to parameters
    assert layer.centers.grad is not None
    assert layer.log_sigma.grad is not None

    # Check for NaN
    assert not torch.isnan(out).any()

def test_glcm_normalization():
    batch_size = 2
    seq_len = 20
    layer = DGLCMLayer(num_bins=4, offsets=[1])
    # Uniform signal
    x = torch.ones(batch_size, seq_len) * 0.5
    out = layer(x)
    assert not torch.isnan(out).any()

    # Constant signal
    x = torch.zeros(batch_size, seq_len)
    out = layer(x)
    assert not torch.isnan(out).any()
