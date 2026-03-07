import torch
from model import FisherDropout

def test_fisher_dropout_update():
    """Tests if the Fisher Information buffer is updated correctly."""
    fd = FisherDropout(10, p_base=0.5, alpha=0.9)
    initial_fisher = fd.fisher_info.clone()

    # Forward pass
    x = torch.randn(5, 10)
    _ = fd(x)

    updated_fisher = fd.fisher_info
    assert not torch.equal(initial_fisher, updated_fisher), "Fisher Info should be updated."

    # Check the update logic: 0.9 * 1.0 + 0.1 * (x**2).mean(0)
    expected_fisher = 0.9 * initial_fisher + 0.1 * (x**2).mean(dim=0)
    assert torch.allclose(updated_fisher, expected_fisher, atol=1e-6), "Fisher update logic incorrect."

def test_fisher_dropout_gamma_positive():
    """Tests if dropout probabilities increase for higher activations when gamma > 0."""
    fd = FisherDropout(10, p_base=0.5, gamma=1.0, alpha=0.0)

    # x has high activation at index 0, others have very small but non-zero activation to avoid mean=0
    x = torch.ones(1, 10) * 0.1
    x[0, 0] = 10.0
    _ = fd(x)

    # This is a bit hard to test precisely without accessing internal p,
    # but we can check if index 0 is more likely to be dropped.
    drops = torch.zeros(10)
    for _ in range(1000):
        out = fd(x)
        drops += (out == 0).float().squeeze()

    assert drops[0] > drops[1], f"Index with high activation should be dropped more with gamma > 0. Got drops: {drops}"

def test_fisher_dropout_gamma_negative():
    """Tests if dropout probabilities decrease for higher activations when gamma < 0."""
    fd = FisherDropout(10, p_base=0.5, gamma=-1.0, alpha=0.0)

    # x has high activation at index 0, others small but non-zero
    x = torch.ones(1, 10) * 0.1
    x[0, 0] = 10.0
    _ = fd(x)

    drops = torch.zeros(10)
    for _ in range(1000):
        out = fd(x)
        drops += (out == 0).float().squeeze()

    assert drops[0] < drops[1], f"Index with high activation should be dropped less with gamma < 0. Got drops: {drops}"
