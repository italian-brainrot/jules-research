import torch
from model import TropicalLinear

def test_tropical_linear_properties():
    in_features = 10
    out_features = 5
    batch_size = 2

    # Test initialization
    layer = TropicalLinear(in_features, out_features, init_beta=1.0)
    x = torch.randn(batch_size, in_features)
    y = layer(x)
    assert y.shape == (batch_size, out_features)

    # Test gradient flow
    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.beta.grad is not None
    print("Gradient flow test passed.")

    # Test large beta behavior (approx max)
    layer_max = TropicalLinear(in_features, out_features, init_beta=100.0)
    with torch.no_grad():
        # Set weights to 0 to test max(x)
        layer_max.weight.fill_(0.0)
        y_max = layer_max(x)
        y_true_max = x.max(dim=1).values
        # All out_features should be approx max(x)
        for i in range(out_features):
            diff = (y_max[:, i] - y_true_max).abs().max()
            assert diff < 1e-1, f"Max approximation failed: {diff}"
    print("Large beta (max) approximation test passed.")

    # Test small beta behavior (approx mean)
    layer_mean = TropicalLinear(in_features, out_features, init_beta=1e-6)
    with torch.no_grad():
        layer_mean.weight.fill_(0.0)
        y_mean = layer_mean(x)
        y_true_mean = x.mean(dim=1)
        for i in range(out_features):
            diff = (y_mean[:, i] - y_true_mean).abs().max()
            assert diff < 1e-3, f"Mean approximation failed: {diff}"
    print("Small beta (mean) approximation test passed.")

    # Test large negative beta behavior (approx min)
    layer_min = TropicalLinear(in_features, out_features, init_beta=-100.0)
    with torch.no_grad():
        layer_min.weight.fill_(0.0)
        y_min = layer_min(x)
        y_true_min = x.min(dim=1).values
        for i in range(out_features):
            diff = (y_min[:, i] - y_true_min).abs().max()
            assert diff < 1e-1, f"Min approximation failed: {diff}"
    print("Large negative beta (min) approximation test passed.")

if __name__ == "__main__":
    test_tropical_linear_properties()
