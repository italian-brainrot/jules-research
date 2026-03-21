import torch
from model import QuadraticLayer, LRQIN

def test_quadratic_layer():
    print("Testing QuadraticLayer...")
    in_features = 40
    out_features = 64
    batch_size = 32

    layer = QuadraticLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features, requires_grad=True)

    # Test shape
    out = layer(x)
    assert out.shape == (batch_size, out_features), f"Expected shape {(batch_size, out_features)}, got {out.shape}"
    print("Shape check passed.")

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to input"
    assert layer.u.grad is not None, "Gradient did not flow back to u"
    assert layer.v.grad is not None, "Gradient did not flow back to v"
    assert layer.linear.weight.grad is not None, "Gradient did not flow back to linear weights"
    print("Gradient flow check passed.")

    # Ensure quadratic term is actually doing something
    # If we set linear part to zero, and u, v to something, output should be quadratic
    with torch.no_grad():
        layer.linear.weight.zero_()
        layer.linear.bias.zero_()
        layer.u.fill_(1.0)
        layer.v.fill_(1.0)

    x_ones = torch.ones(1, in_features)
    out_ones = layer(x_ones)
    # Ux = sum(x_i) = in_features
    # Vx = sum(x_i) = in_features
    # quad = Ux * Vx = in_features^2
    expected_val = float(in_features**2)
    assert torch.allclose(out_ones, torch.tensor([expected_val] * out_features)), f"Expected {expected_val}, got {out_ones[0,0]}"
    print("Quadratic logic check passed.")

def test_lrqin():
    print("\nTesting LRQIN...")
    model = LRQIN()
    x = torch.randn(16, 40)
    out = model(x)
    assert out.shape == (16, 10), f"Expected shape (16, 10), got {out.shape}"
    print("LRQIN shape check passed.")

if __name__ == "__main__":
    test_quadratic_layer()
    test_lrqin()
    print("\nAll tests passed!")
