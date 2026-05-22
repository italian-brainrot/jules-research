import torch
from model import DLDELayer

def test_gradient_tau():
    B, L, m = 2, 40, 5
    x = torch.randn(B, L, requires_grad=False)
    layer = DLDELayer(m=m, L=L, initial_tau=2.5)

    # Check if tau has gradient
    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    if layer.tau.grad is not None:
        print(f"Gradient of tau: {layer.tau.grad.item()}")
        assert layer.tau.grad != 0, "Gradient of tau is zero!"
    else:
        print("Gradient of tau is None!")
        assert False, "Gradient of tau is None!"

def test_interpolation_logic():
    # Simple signal: [0, 1, 2, 3, 4]
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    L = 5
    m = 2
    # tau = 0.5. indices: j=0..4, d=0..1
    # d=0: 0, 1, 2, 3, 4
    # d=1: -0.5, 0.5, 1.5, 2.5, 3.5
    # Clamped -0.5 -> 0. x[0]=0.0. alpha = -0.5 - floor(-1) = 0.5.
    # Wait, floor(-0.5) = -1. low = -1, high = 0. clamped: low=0, high=0.
    # Actually floor(-0.5) is -1.
    # Let's check tau=1.5
    # d=1: -1.5, -0.5, 0.5, 1.5, 2.5

    layer = DLDELayer(m=m, L=L, initial_tau=1.5)
    out = layer(x) # (1, 2, 5)

    print("Signal:", x)
    print("Tau:", layer.tau.item())
    print("Output:\n", out)

    # At index j=2, d=1: index = 2 - 1 * 1.5 = 0.5
    # x[0]=0.0, x[1]=1.0. interp: 0.0 + 0.5 * (1.0 - 0.0) = 0.5
    assert torch.allclose(out[0, 1, 2], torch.tensor(0.5)), f"Expected 0.5, got {out[0, 1, 2]}"

if __name__ == "__main__":
    print("Testing gradient of tau...")
    test_gradient_tau()
    print("Testing interpolation logic...")
    test_interpolation_logic()
    print("All tests passed!")
