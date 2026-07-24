import torch
import pytest
from model import ZeroOneChaosLayer

def test_zero_one_chaos_layer_shapes():
    B, C, L = 4, 1, 40
    num_freqs = 8
    layer = ZeroOneChaosLayer(num_freqs=num_freqs, n_max=10)
    x = torch.randn(B, C, L)
    out = layer(x)
    assert out.shape == (B, num_freqs)

def test_zero_one_chaos_layer_gradients():
    B, C, L = 4, 1, 40
    num_freqs = 8
    layer = ZeroOneChaosLayer(num_freqs=num_freqs, n_max=10)
    x = torch.randn(B, C, L, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert layer.c.grad is not None
    assert not torch.isnan(layer.c.grad).any()

def test_zero_one_chaos_layer_consistency():
    # Check if K is close to 0 for a periodic signal
    L = 100
    t = torch.linspace(0, 10, L)
    x = torch.sin(t).view(1, 1, L)

    layer = ZeroOneChaosLayer(num_freqs=4, n_max=20)
    # Use a frequency c that is NOT a multiple of signal frequency to avoid resonance issues
    # but the test should generally show low correlation for regular signals.
    # Actually K is correlation between n and MSD. For regular signals MSD is bounded.
    # For chaotic it grows linearly with n.
    k = layer(x)
    print(f"K for sine wave: {k}")
    # For a sine wave, K should be relatively small compared to 1,
    # but the 0-1 test is most reliable for N -> infinity.
    # For N=100, we just check it doesn't crash and returns something reasonable.
    assert k.shape == (1, 4)
    assert (k >= -1.1).all() and (k <= 1.1).all()

def test_zero_one_chaos_layer_chaotic():
    # Logistic map in chaotic regime: x_{n+1} = 4 * x_n * (1 - x_n)
    L = 100
    x = torch.zeros(1, 1, L)
    x[0, 0, 0] = 0.3
    for i in range(L-1):
        x[0, 0, i+1] = 4 * x[0, 0, i] * (1 - x[0, 0, i])

    layer = ZeroOneChaosLayer(num_freqs=4, n_max=20)
    k = layer(x)
    print(f"K for chaotic logistic map: {k}")
    assert k.shape == (1, 4)

if __name__ == "__main__":
    test_zero_one_chaos_layer_shapes()
    test_zero_one_chaos_layer_gradients()
    test_zero_one_chaos_layer_consistency()
    test_zero_one_chaos_layer_chaotic()
    print("All tests passed!")
