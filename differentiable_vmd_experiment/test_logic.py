import torch
import pytest
from vmd import DVMD

def test_vmd_gradients():
    batch_size = 2
    T = 40
    n_modes = 2
    vmd = DVMD(n_modes=n_modes, n_iter=10)

    x = torch.randn(batch_size, T, requires_grad=True)
    energies, omegas = vmd(x)

    loss = energies.sum() + omegas.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()

    assert vmd.alpha.grad is not None
    assert not torch.isnan(vmd.alpha.grad).any()

def test_vmd_frequency_detection():
    T = 64
    t = torch.linspace(0, 1, T)
    # Frequencies 10Hz and 20Hz
    # Normalized frequencies: 10/64 = 0.15625, 20/64 = 0.3125
    x = torch.sin(2 * 3.14159 * 10 * t) + torch.sin(2 * 3.14159 * 20 * t)
    x = x.unsqueeze(0)

    vmd = DVMD(n_modes=2, n_iter=100)
    energies, omegas = vmd(x)

    omegas_np = omegas.detach().cpu().numpy()[0]
    expected_omegas = [10/64, 20/64]

    print(f"Detected omegas: {omegas_np}")
    print(f"Expected omegas: {expected_omegas}")

    for i in range(2):
        assert abs(omegas_np[i] - expected_omegas[i]) < 0.05

if __name__ == "__main__":
    test_vmd_gradients()
    test_vmd_frequency_detection()
    print("Tests passed!")
