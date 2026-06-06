import torch
from model import DCDLayer

def test_dcd_differentiability():
    batch_size = 4
    seq_len = 40
    x = torch.randn(batch_size, seq_len, requires_grad=True)
    layer = DCDLayer(m=3, tau=2, gamma=10.0)

    d2, cr = layer(x)

    assert d2.shape == (batch_size, 1)
    assert cr.shape == (batch_size, 8)

    loss = d2.sum() + cr.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0).any()
    print("Differentiability test passed!")

def test_dcd_scaling():
    # Constant signal should have low correlation dimension
    x = torch.zeros(2, 40) + torch.randn(2, 40) * 0.001
    layer = DCDLayer(m=2, tau=1)
    d2_flat, _ = layer(x)

    # White noise should have higher correlation dimension
    x_noise = torch.randn(2, 40)
    d2_noise, _ = layer(x_noise)

    print(f"D2 (flat): {d2_flat.mean().item():.4f}")
    print(f"D2 (noise): {d2_noise.mean().item():.4f}")

    # Noise should generally have higher dimension than a flat signal
    # although N=40 is small for accurate D2 estimation, it should show a trend
    assert d2_noise.mean() > d2_flat.mean()
    print("Scaling test passed!")

def test_gamma_grad():
    x = torch.randn(4, 40)
    layer = DCDLayer(m=3, tau=1, learnable_params=True)
    d2, cr = layer(x)
    loss = d2.sum()
    loss.backward()

    assert layer.gamma.grad is not None
    assert layer.gamma.grad != 0
    print("Gamma gradient test passed!")

if __name__ == "__main__":
    test_dcd_differentiability()
    test_dcd_scaling()
    test_gamma_grad()
