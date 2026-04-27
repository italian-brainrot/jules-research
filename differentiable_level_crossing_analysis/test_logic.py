import torch
from dlca import DLCALayer

def test_dlca_logic():
    batch_size = 2
    seq_len = 100
    num_levels = 5
    layer = DLCALayer(num_levels=num_levels, beta=100.0) # High beta for sharp approximation

    # Create a sine wave that crosses 0 multiple times
    t = torch.linspace(0, 4 * 3.14159, seq_len)
    x = torch.sin(t).unsqueeze(0).repeat(batch_size, 1)

    features = layer(x)
    print(f"Features shape: {features.shape}")

    # 4*pi means 2 full periods.
    # sin(t) starts at 0, goes up to 1, down to -1, back to 0. (Period 1)
    # Then repeats. (Period 2)
    # Level 0.0:
    # Upward crossings: at t=0 (depending on next), t=2*pi, t=4*pi
    # Downward crossings: at t=pi, t=3*pi

    # Let's check if we get non-zero features
    assert features.shape == (batch_size, 2 * num_levels)
    assert torch.all(features >= 0)
    print("Logic test passed!")

def test_dlca_differentiability():
    batch_size = 2
    seq_len = 10
    num_levels = 3
    layer = DLCALayer(num_levels=num_levels, beta=10.0)

    x = torch.randn(batch_size, seq_len, requires_grad=True)
    features = layer(x)

    loss = features.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Check gradients for levels and beta
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            print(f"Gradient for {name} is OK")

    print("Differentiability test passed!")

if __name__ == "__main__":
    test_dlca_logic()
    test_dlca_differentiability()
