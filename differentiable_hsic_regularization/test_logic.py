import torch
import pytest
from differentiable_hsic_regularization.hsic import hsic, hsic_normalized

def test_hsic_logic():
    n = 100
    d1 = 10
    d2 = 5

    # Dependent data: y is a function of x
    x = torch.randn(n, d1)
    y = x[:, :d2] ** 2

    h_val_dep = hsic(x, y)
    print(f"HSIC (dependent): {h_val_dep.item()}")

    # Independent data
    x_ind = torch.randn(n, d1)
    y_ind = torch.randn(n, d2)

    h_val_ind = hsic(x_ind, y_ind)
    print(f"HSIC (independent): {h_val_ind.item()}")

    assert h_val_dep > h_val_ind

def test_hsic_gradients():
    n = 10
    d1 = 5
    d2 = 5

    x = torch.randn(n, d1, requires_grad=True)
    y = torch.randn(n, d2, requires_grad=True)

    h_val = hsic(x, y)
    h_val.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert not torch.all(x.grad == 0)
    assert not torch.all(y.grad == 0)

def test_hsic_normalized_logic():
    n = 100
    x = torch.randn(n, 5)
    y = x * 2

    h_norm = hsic_normalized(x, y)
    print(f"Normalized HSIC (same): {h_norm.item()}")
    assert h_norm > 0.5

    y_rand = torch.randn(n, 5)
    h_norm_rand = hsic_normalized(x, y_rand)
    print(f"Normalized HSIC (rand): {h_norm_rand.item()}")
    assert h_norm > h_norm_rand

if __name__ == "__main__":
    test_hsic_logic()
    test_hsic_gradients()
    test_hsic_normalized_logic()
    print("All tests passed!")
