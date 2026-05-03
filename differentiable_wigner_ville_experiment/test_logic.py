import torch
from model import WignerVilleLayer

def test_wvd_shape():
    batch_size = 4
    seq_len = 40
    n_fft = 32
    window_size = 15
    layer = WignerVilleLayer(n_fft=n_fft, window_size=window_size)
    x = torch.randn(batch_size, seq_len)
    output = layer(x)

    assert output.shape == (batch_size, seq_len, n_fft)
    print(f"Shape test passed: {output.shape}")

def test_wvd_differentiability():
    batch_size = 2
    seq_len = 20
    n_fft = 16
    window_size = 11
    layer = WignerVilleLayer(n_fft=n_fft, window_size=window_size)
    x = torch.randn(batch_size, seq_len, requires_grad=True)
    output = layer(x)
    loss = output.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Differentiability test passed.")

def test_wvd_real_output():
    batch_size = 2
    seq_len = 20
    layer = WignerVilleLayer()
    x = torch.randn(batch_size, seq_len)
    output = layer(x)

    assert torch.is_floating_point(output)
    print("Real output test passed.")

if __name__ == "__main__":
    test_wvd_shape()
    test_wvd_differentiability()
    test_wvd_real_output()
