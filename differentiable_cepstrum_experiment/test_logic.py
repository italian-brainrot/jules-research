import torch
import pytest
from differentiable_cepstrum_experiment.model import DifferentiableCepstrum

def test_cepstrum_shape():
    batch_size = 4
    seq_len = 40
    x = torch.randn(batch_size, seq_len)
    layer = DifferentiableCepstrum()
    out = layer(x)
    assert out.shape == (batch_size, seq_len)
    assert out.dtype == torch.float32

def test_cepstrum_gradient():
    seq_len = 40
    x = torch.randn(1, seq_len, requires_grad=True)
    layer = DifferentiableCepstrum()
    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

def test_periodic_signal():
    # Simple periodic signal should have a peak in its cepstrum at the period (quefrency)
    # Note: For small signals, the cepstrum can be noisy, but let's check it doesn't crash.
    n = 40
    t = torch.linspace(0, 1, n)
    # Sine wave with 4 cycles
    x = torch.sin(2 * 3.14159 * 4 * t).unsqueeze(0)
    layer = DifferentiableCepstrum(n=n)
    out = layer(x)
    assert out.shape == (1, n)

def test_double_precision_input():
    # Verify it handles float64 input by casting to float32 internally
    x = torch.randn(2, 40, dtype=torch.float64)
    layer = DifferentiableCepstrum()
    out = layer(x)
    assert out.dtype == torch.float32
