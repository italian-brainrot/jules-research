import torch
from differentiable_fisher_vector_experiment.model import FisherVectorLayer

def test_fisher_vector_layer():
    batch_size = 4
    length = 40
    num_clusters = 5
    patch_size = 10
    stride = 5

    layer = FisherVectorLayer(num_clusters, patch_size, stride)
    x = torch.randn(batch_size, 1, length, requires_grad=True)

    # Forward pass
    fv = layer(x)

    # Check output shape
    # u: K * patch_size, v: K * patch_size
    # Total dim: 2 * K * patch_size = 2 * 5 * 10 = 100
    expected_dim = 2 * num_clusters * patch_size
    assert fv.shape == (batch_size, expected_dim)
    print(f"Output shape: {fv.shape} (Expected: {(batch_size, expected_dim)})")

    # Check differentiability
    loss = fv.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.logits.grad is not None
    assert layer.means.grad is not None
    assert layer.log_vars.grad is not None

    print("Gradients with respect to input and parameters are present.")

    # Check normalization
    norm = torch.norm(fv, p=2, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    print("L2 normalization is correct.")

if __name__ == "__main__":
    test_fisher_vector_layer()
