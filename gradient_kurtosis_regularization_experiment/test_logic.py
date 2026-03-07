import torch
import torch.nn as nn
from gradient_kurtosis_regularization_experiment.gkr_lib import compute_gradient_kurtosis, compute_per_sample_grad_norms
import pytest

def test_kurtosis():
    # Symmetric, uniform distribution kurtosis is around -1.2
    # Normal distribution kurtosis is 3.0 (excess 0.0)
    # Laplace distribution is peaked, excess kurtosis is 3.0

    # Test Normal: should be near 0
    torch.manual_seed(42)
    norms_normal = torch.randn(10000)
    kurt_normal = compute_gradient_kurtosis(norms_normal)
    print(f"Normal kurtosis: {kurt_normal.item()}")
    assert abs(kurt_normal.item()) < 0.2

    # Test highly peaked: should be > 0
    # For a high kurtosis example, take 999 samples from N(0, 1) and 1 sample from N(0, 100)
    norms_peaked = torch.randn(1000)
    norms_peaked[-1] = 20.0 # Extreme outlier
    kurt_peaked = compute_gradient_kurtosis(norms_peaked)
    print(f"Peaked kurtosis: {kurt_peaked.item()}")
    assert kurt_peaked.item() > 5.0

    # Test Uniform: should be < 0
    # Uniform has excess kurtosis -1.2
    norms_uniform = torch.rand(10000)
    kurt_uniform = compute_gradient_kurtosis(norms_uniform)
    print(f"Uniform kurtosis: {kurt_uniform.item()}")
    assert kurt_uniform.item() < -1.0

def test_grad_norms():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))

    norms = compute_per_sample_grad_norms(model, x, y)
    print(f"Norms: {norms}")
    assert norms.shape[0] == 5
    assert (norms >= 0).all()

if __name__ == "__main__":
    test_kurtosis()
    test_grad_norms()
    print("Tests passed!")
