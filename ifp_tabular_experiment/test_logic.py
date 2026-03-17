import torch
import pytest
from ifp_tabular_experiment.model import RIFPLayer

def test_rifp_independence():
    num_features = 10
    hidden_dim = 8
    batch_size = 4
    model = RIFPLayer(num_features, hidden_dim)

    # Initial input
    x = torch.randn(batch_size, num_features)
    x.requires_grad = True

    # Get output
    y = model(x)

    # Check each output feature for independence
    for i in range(num_features):
        # Gradient of the i-th output feature with respect to all input features
        grad = torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0]

        # For the i-th output feature, only the i-th input feature's gradient should be non-zero
        for j in range(num_features):
            if i == j:
                assert (grad[:, j] != 0).any(), f"Output feature {i} should depend on input feature {j}"
            else:
                assert (grad[:, j] == 0).all(), f"Output feature {i} should NOT depend on input feature {j}"

def test_rifp_shape():
    num_features = 40
    batch_size = 16
    model = RIFPLayer(num_features)
    x = torch.randn(batch_size, num_features)
    y = model(x)
    assert y.shape == (batch_size, num_features)

if __name__ == "__main__":
    test_rifp_independence()
    test_rifp_shape()
    print("Tests passed!")
