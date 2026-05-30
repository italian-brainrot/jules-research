import torch
from model import DHVGLayer

def test_dhvg_shape():
    L = 20
    batch_size = 4
    layer = DHVGLayer(L=L)
    x = torch.randn(batch_size, L)
    A = layer(x)
    assert A.shape == (batch_size, L, L)
    # Check symmetry
    assert torch.allclose(A, A.transpose(1, 2), atol=1e-6)

def test_dhvg_differentiability():
    L = 20
    batch_size = 2
    layer = DHVGLayer(L=L)
    x = torch.randn(batch_size, L, requires_grad=True)
    A = layer(x)
    loss = A.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.scale.grad is not None

def test_dhvg_logic():
    # Simple case: x = [1, 0, 1]
    # min(x_0, x_2) = 1. x_1 = 0.
    # So 0 and 2 should be connected because x_1 < min(x_0, x_2)
    L = 3
    layer = DHVGLayer(L=L, initial_scale=100.0) # High scale for hard threshold
    x = torch.tensor([[1.0, 0.0, 1.0]])
    A = layer(x)
    # A should be roughly [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # Adjacent are always visible. 0-2 visibility depends on x1 < min(x0, x2).
    expected = torch.tensor([[[0.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0],
                             [1.0, 1.0, 0.0]]])
    assert torch.allclose(A, expected, atol=1e-2)

    # Case: x = [0, 1, 0]
    # min(x_0, x_2) = 0. x_1 = 1.
    # x_1 is NOT < min(x_0, x_2). So 0 and 2 should NOT be connected.
    x2 = torch.tensor([[0.0, 1.0, 0.0]])
    A2 = layer(x2)
    expected2 = torch.tensor([[[0.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0]]])
    assert torch.allclose(A2, expected2, atol=1e-2)

if __name__ == "__main__":
    test_dhvg_shape()
    test_dhvg_differentiability()
    test_dhvg_logic()
    print("All logic tests passed!")
