import torch
from ssa import DifferentiableSSA

def test_ssa_gradient_flow():
    batch_size = 4
    input_len = 40
    window_size = 10

    x = torch.randn(batch_size, input_len, requires_grad=True)
    ssa = DifferentiableSSA(window_size)

    out = ssa(x)

    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"

    loss = out.pow(2).sum()
    loss.backward()

    assert x.grad is not None, "Gradient w.r.t input is None"
    assert ssa.weights.grad is not None, "Gradient w.r.t weights is None"
    print("Gradient flow test passed.")

def test_ssa_reconstruction():
    # If weights are all 1, SSA (with full rank) should be identity?
    # Not necessarily exactly, but it should be close if rank is max.
    # Actually, SSA with full rank should perfectly reconstruct the Hankel matrix,
    # and diagonal averaging should then reconstruct the original signal.

    batch_size = 2
    input_len = 20
    window_size = 5

    x = torch.randn(batch_size, input_len)
    ssa = DifferentiableSSA(window_size)

    # Set weights high so sigmoid(weights) is close to 1
    with torch.no_grad():
        ssa.weights.fill_(20.0)

    out = ssa(x)

    diff = (x - out).abs().max().item()
    print(f"Reconstruction error (max diff): {diff:.6e}")
    assert diff < 1e-4, f"Reconstruction failed, diff: {diff}"
    print("Reconstruction test passed.")

if __name__ == "__main__":
    test_ssa_gradient_flow()
    test_ssa_reconstruction()
