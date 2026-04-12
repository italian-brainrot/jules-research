import torch
from model import DOPLayer

def test_dop_logic():
    B, C, L = 2, 1, 10
    d, tau = 3, 2
    layer = DOPLayer(d=d, tau=tau)

    x = torch.randn(B, C, L, requires_grad=True)
    out = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Expected num_windows: (L - ((d-1)*tau + 1)) + 1 = (10 - (2*2 + 1)) + 1 = (10 - 5) + 1 = 6
    # Expected num_pairs: d*(d-1)/2 = 3*2/2 = 3
    # Expected output shape: (2, 1*3, 6)

    assert out.shape == (B, C * (d * (d - 1) // 2), 6)

    # Gradient check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    print("Gradient flow check passed.")

def test_scale_invariance():
    B, C, L = 1, 1, 5
    d, tau = 3, 1
    layer = DOPLayer(d=d, tau=tau, temperature=100.0) # High temperature for near-hard comparisons

    x1 = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    x2 = x1 * 10.0
    x3 = x1 + 5.0

    out1 = layer(x1)
    out2 = layer(x2)
    out3 = layer(x3)

    # With high temperature, they should be very similar
    diff2 = torch.abs(out1 - out2).max().item()
    diff3 = torch.abs(out1 - out3).max().item()

    print(f"Max diff (scaled): {diff2}")
    print(f"Max diff (shifted): {diff3}")

    assert diff2 < 1e-3
    assert diff3 < 1e-3
    print("Scale and shift invariance (approximate) check passed.")

if __name__ == "__main__":
    test_dop_logic()
    test_scale_invariance()
