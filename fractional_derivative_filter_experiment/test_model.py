import torch
import math
from model import FractionalFilterbank

def test_fractional_filterbank_identity():
    # When alpha is 0 and gain is 1, it should be identity (with padding/cropping)
    # Actually, we zero out DC, so it won't be perfect identity unless we add DC back.
    # But let's check if the signal shape and some properties are preserved.
    in_channels = 1
    out_channels = 1
    n = 40
    layer = FractionalFilterbank(in_channels, out_channels, init_alpha=0.1)

    # Manually set alpha to 0 and gain to 1, bias to 0
    # sigmoid(raw)*2 = 0 -> raw = -inf. We'll use a very small value.
    layer.alpha_raw.data.fill_(-100.0)
    layer.gain.data.fill_(1.0)
    layer.bias.data.fill_(0.0)

    x = torch.randn(2, in_channels, n)
    # Remove mean to avoid DC zeroing issue in test
    x = x - x.mean(dim=-1, keepdim=True)

    y = layer(x)

    assert y.shape == (2, in_channels * out_channels, n)
    # Check if y is close to x (since alpha=0 and DC is removed but we removed it from x too)
    # Note: padding and FFT might introduce small errors
    diff = torch.abs(x - y).max().item()
    print(f"Max diff: {diff}")
    assert diff < 1e-5

if __name__ == "__main__":
    test_fractional_filterbank_identity()
    print("Test passed!")
