import torch
from model import DifferentiableDFALayer

def test_dfa_differentiability():
    batch_size = 4
    signal_length = 40
    scales = [4, 8, 10]

    x = torch.randn(batch_size, signal_length, requires_grad=True)
    layer = DifferentiableDFALayer(scales=scales)

    features = layer(x)

    # Check output shape
    # scales (3) + alpha (1) = 4
    expected_dim = len(scales) + 1
    assert features.shape == (batch_size, expected_dim)

    # Check differentiability
    loss = features.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Differentiability test passed!")

def test_dfa_scaling():
    # White noise should have alpha around 0.5
    # Brownian noise (integral of white noise) should have alpha around 1.5
    # Pink noise should have alpha around 1.0

    signal_length = 1000
    scales = [10, 20, 40, 80, 100]
    layer = DifferentiableDFALayer(scales=scales)

    # White noise
    torch.manual_seed(42)
    x_white = torch.randn(10, signal_length)
    features_white = layer(x_white)
    alpha_white = features_white[:, -1].mean().item()
    print(f"Alpha for white noise: {alpha_white:.4f} (expected around 0.5)")

    # Brownian noise (integrated white noise)
    x_brownian = torch.cumsum(torch.randn(10, signal_length), dim=-1)
    features_brownian = layer(x_brownian)
    alpha_brownian = features_brownian[:, -1].mean().item()
    print(f"Alpha for Brownian noise: {alpha_brownian:.4f} (expected around 1.5)")

if __name__ == "__main__":
    test_dfa_differentiability()
    test_dfa_scaling()
