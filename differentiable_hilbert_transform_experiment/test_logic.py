import torch
import numpy as np
from differentiable_hilbert_transform_experiment.model import DifferentiableHilbertLayer

def test_hilbert_shapes():
    batch_size, channels, length = 8, 2, 40
    layer = DifferentiableHilbertLayer()
    x = torch.randn(batch_size, channels, length)
    y = layer(x)
    assert y.shape == (batch_size, 3 * channels, length)

def test_hilbert_differentiability():
    batch_size, channels, length = 4, 1, 32
    layer = DifferentiableHilbertLayer()
    x = torch.randn(batch_size, channels, length, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

def test_hilbert_envelope():
    # A simple sine wave should have a constant envelope (approximately)
    length = 128
    t = torch.linspace(0, 2 * np.pi * 10, length)
    x = torch.sin(t).view(1, 1, length)

    layer = DifferentiableHilbertLayer()
    y = layer(x)

    # Original signal is y[:, 0, :]
    # Envelope is y[:, 1, :]
    # Frequency is y[:, 2, :]

    envelope = y[0, 1, :].detach().numpy()

    # Envelope of sin(t) is 1.0.
    # Because of finite length and edge effects, it might not be exactly 1.0 at edges.
    # We check the middle part.
    mid_envelope = envelope[10:-10]
    assert np.allclose(mid_envelope, 1.0, atol=0.1)

def test_hilbert_frequency():
    # For sin(omega * t), instantaneous frequency should be constant omega
    length = 256
    omega = 0.5 # radians per sample
    t = torch.arange(length).float()
    x = torch.sin(omega * t).view(1, 1, length)

    layer = DifferentiableHilbertLayer()
    y = layer(x)

    # Frequency is y[:, 2, :]
    frequency = y[0, 2, :].detach().numpy()

    # Note: our frequency estimation is angle(z[n] * conj(z[n-1]))
    # For e^{i omega n}, z[n] * conj(z[n-1]) = e^{i omega n} * e^{-i omega (n-1)} = e^{i omega}
    # angle(e^{i omega}) = omega

    # Check middle part to avoid edge effects
    mid_frequency = frequency[10:-10]
    assert np.allclose(mid_frequency, omega, atol=0.1)
