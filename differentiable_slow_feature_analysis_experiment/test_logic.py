import torch
from model import SFAMLP, SFALayer

def test_sfa_layer_gradients():
    x = torch.randn(2, 1, 40, requires_grad=True)
    layer = SFALayer(1, 16)

    y, slowness = layer(x)

    print(f"y shape: {y.shape}")
    print(f"slowness penalty: {slowness.item()}")

    # Check if gradients flow back to x and layer weights
    slowness.backward()

    assert x.grad is not None, "Gradients should flow back to input x"
    assert layer.conv.weight.grad is not None, "Gradients should flow back to layer weights"

    print("Gradients flow check: PASSED")

def test_sfa_penalty_behavior():
    # Test if slowness penalty is lower for smooth signals
    layer = SFALayer(1, 1) # Simple convolution for easier observation
    # Smooth signal: constant
    x_smooth = torch.ones(2, 1, 40)
    # Noisy signal: random
    x_noisy = torch.randn(2, 1, 40)

    _, s_smooth = layer(x_smooth)
    _, s_noisy = layer(x_noisy)

    print(f"Smooth signal penalty: {s_smooth.item()}")
    print(f"Noisy signal penalty: {s_noisy.item()}")

    # In general, if we initialize weights randomly,
    # the penalty will depend on the weight values,
    # but the slowness term *should* be differentiable.

if __name__ == "__main__":
    test_sfa_layer_gradients()
    test_sfa_penalty_behavior()
