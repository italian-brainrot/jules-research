import torch
from model import Differentiable1DPersistence

def test_persistence_logic():
    # Simple signal with two minima and one maximum in between
    # x = [1, 0, 1, 2, 1, 3]
    # Minima at index 1 (val 0), index 4 (val 1)
    # Maximum between them is at index 3 (val 2)
    # Pair 1: birth=1, death=2 (persistence 1)
    # The other minimum is global, doesn't die in this logic (or dies at infinity)
    # Our logic gives death at boundary/max.

    layer = Differentiable1DPersistence(top_k=2)
    x = torch.tensor([[1.0, 0.0, 1.0, 2.0, 1.0, 3.0]], requires_grad=True)
    p = layer(x)
    print("Persistence:", p)

    assert p.shape == (1, 2)
    assert p[0, 0] > 0

    # Check differentiability
    loss = p.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)
    print("Gradient:", x.grad)

def test_persistence_batch():
    layer = Differentiable1DPersistence(top_k=5)
    x = torch.randn(4, 40, requires_grad=True)
    p = layer(x)
    assert p.shape == (4, 5)

    loss = p.mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape

if __name__ == "__main__":
    test_persistence_logic()
    test_persistence_batch()
    print("All tests passed!")
