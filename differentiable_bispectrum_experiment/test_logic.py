import torch
from bispectrum import DifferentiableBispectrum

def test_bispectrum_differentiability():
    batch_size = 4
    n_input = 40
    x = torch.randn(batch_size, n_input, requires_grad=True)

    layer = DifferentiableBispectrum(n_input, use_magnitude=True, use_phase=True)
    y = layer(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Differentiability test passed!")

def test_bispectrum_shape():
    batch_size = 4
    n_input = 40
    x = torch.randn(batch_size, n_input)

    layer = DifferentiableBispectrum(n_input, use_magnitude=True, use_phase=False)
    y = layer(x)
    print(f"Output shape (magnitude only): {y.shape}")

    layer_both = DifferentiableBispectrum(n_input, use_magnitude=True, use_phase=True)
    y_both = layer_both(x)
    print(f"Output shape (magnitude and phase): {y_both.shape}")

    assert y_both.shape[-1] == 2 * y.shape[-1]
    print("Shape test passed!")

def test_bispectrum_invariance():
    # Bispectrum of real signals is shift-invariant in theory (the magnitude is, and phase is modified linearly but often we use it carefully)
    # Actually, the magnitude of the bispectrum |B(k, l)| IS shift-invariant.
    n_input = 40
    x = torch.randn(1, n_input)

    # Roll the input
    x_shifted = torch.roll(x, shifts=5, dims=1)

    layer = DifferentiableBispectrum(n_input, use_magnitude=True, use_phase=False)

    y = layer(x)
    y_shifted = layer(x_shifted)

    diff = torch.abs(y - y_shifted).max().item()
    rel_diff = (torch.abs(y - y_shifted) / (torch.abs(y) + 1e-8)).max().item()
    print(f"Max difference after shift: {diff}")
    print(f"Max relative difference: {rel_diff}")
    # It might not be EXACTLY zero due to how RFFT treats shifted signals if not circular,
    # but torch.roll is circular.
    assert rel_diff < 1e-4
    print("Shift invariance test passed!")

if __name__ == "__main__":
    test_bispectrum_differentiability()
    test_bispectrum_shape()
    test_bispectrum_invariance()
