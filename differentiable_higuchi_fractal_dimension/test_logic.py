import torch
from higuchi import HiguchiFractalDimension, compute_hfd

def test_higuchi_fractal_dimension():
    B, N = 4, 40
    k_max = 10
    x = torch.randn(B, N, requires_grad=True)

    hfd_layer = HiguchiFractalDimension(k_max=k_max)
    L_k = hfd_layer(x)

    print(f"Output shape: {L_k.shape}")
    assert L_k.shape == (B, k_max)
    assert torch.all(L_k > 0)

    # Check gradients
    loss = L_k.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Gradients for L_k are OK.")

def test_compute_hfd():
    B, N = 2, 100
    # Generate some simple signals
    # Sine wave (should have low fractal dimension)
    t = torch.linspace(0, 10, N)
    x1 = torch.sin(t)
    # Brownian motion (should have higher fractal dimension ~1.5)
    x2 = torch.cumsum(torch.randn(N), dim=0)

    x = torch.stack([x1, x2])
    D = compute_hfd(x, k_max=20)
    print(f"HFD for sine wave: {D[0].item():.4f}")
    print(f"HFD for Brownian motion: {D[1].item():.4f}")

    assert D[1] > D[0]

    # Gradient check for D
    x_grad = torch.tensor(x, requires_grad=True)
    D_grad = compute_hfd(x_grad, k_max=20)
    D_grad.sum().backward()
    assert x_grad.grad is not None
    print("Gradients for D are OK.")

if __name__ == "__main__":
    test_higuchi_fractal_dimension()
    test_compute_hfd()
