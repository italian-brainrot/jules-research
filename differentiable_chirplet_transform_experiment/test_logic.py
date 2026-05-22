import torch
from model import ChirpletLayer

def test_chirplet_differentiability():
    batch_size = 2
    in_channels = 1
    out_channels = 4
    kernel_size = 15
    signal_length = 40

    layer = ChirpletLayer(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    x = torch.randn(batch_size, in_channels, signal_length, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradient w.r.t input is None"
    assert layer.frequencies.grad is not None, "Gradient w.r.t frequencies is None"
    assert layer.sigmas.grad is not None, "Gradient w.r.t sigmas is None"
    assert layer.phases.grad is not None, "Gradient w.r.t phases is None"
    assert layer.chirp_rates.grad is not None, "Gradient w.r.t chirp_rates is None"

    print("Differentiability test passed!")

def test_chirplet_shapes():
    batch_size = 2
    in_channels = 1
    out_channels = 8
    kernel_size = 15
    signal_length = 40

    layer = ChirpletLayer(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    x = torch.randn(batch_size, in_channels, signal_length)

    output = layer(x)
    assert output.shape == (batch_size, out_channels, signal_length), f"Expected shape {(batch_size, out_channels, signal_length)}, got {output.shape}"
    print("Shape test passed!")

if __name__ == "__main__":
    test_chirplet_differentiability()
    test_chirplet_shapes()
