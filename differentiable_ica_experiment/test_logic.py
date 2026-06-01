import torch
import pytest
from differentiable_ica_experiment.model import DICALayer, PCALayer

def test_ica_layer_differentiability():
    batch_size = 100
    num_features = 10
    num_components = 10

    x = torch.randn(batch_size, num_features, requires_grad=True)
    ica = DICALayer(num_features, num_components, iterations=5)

    output = ica(x)
    loss = output.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("DICA Layer gradient check passed.")

def test_pca_layer_differentiability():
    batch_size = 100
    num_features = 10
    num_components = 5

    x = torch.randn(batch_size, num_features, requires_grad=True)
    pca = PCALayer(num_features, num_components)

    output = pca(x)
    loss = output.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("PCA Layer gradient check passed.")

def test_ica_orthogonality():
    batch_size = 200
    num_features = 5
    num_components = 5

    # Generate some non-Gaussian data (independent components)
    s = torch.sign(torch.randn(batch_size, num_components)) # Rademacher
    A = torch.randn(num_components, num_features)
    x = s @ A # Mixed signals

    ica = DICALayer(num_features, num_components, iterations=20)
    output = ica(x) # (B, num_components)

    # The output components should be approximately uncorrelated (whitened)
    # Since FastICA uses whitening and symmetric decorrelation,
    # Cov(output) should be Identity.
    cov = (output.t() @ output) / (batch_size - 1)

    # We ignore the very small diagonal eps if any, but since we subtract mean and whiten:
    identity = torch.eye(num_components)
    # FastICA with few iterations and finite samples might not be perfectly identity,
    # but should be very close. Increasing tolerance or iterations.
    torch.testing.assert_close(cov, identity, atol=1e-2, rtol=1e-2)
    print("DICA Layer orthogonality/whitening check passed.")

if __name__ == "__main__":
    test_ica_layer_differentiability()
    test_pca_layer_differentiability()
    test_ica_orthogonality()
