import torch
import torch.nn as nn
from burg_method import burg_method, get_autocorrelation, levinson_recursion

def test_burg_gradient():
    x = torch.randn(4, 40, requires_grad=True)
    order = 5
    a = burg_method(x, order)
    assert a.shape == (4, order)

    loss = a.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Burg gradient test passed")

def test_levinson_gradient():
    x = torch.randn(4, 40, requires_grad=True)
    order = 5
    r = get_autocorrelation(x, order)
    a = levinson_recursion(r, order)
    assert a.shape == (4, order)

    loss = a.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("Levinson gradient test passed")

def test_burg_values():
    # Test on a simple autoregressive signal
    # x_t = 0.5 * x_{t-1} + noise
    length = 100
    x = torch.zeros(1, length)
    for t in range(1, length):
        x[0, t] = 0.5 * x[0, t-1] + torch.randn(1) * 0.1

    order = 1
    a = burg_method(x, order)
    print(f"LPC coeff for 0.5 AR process (Burg): {a.item()}")
    # should be close to 0.5 (or -0.5 depending on sign convention)
    # wait, my burg returns a_i where x_hat[t] = sum a_i x[t-i]
    # let's check.

    a_lev = levinson_recursion(get_autocorrelation(x, order), order)
    print(f"LPC coeff for 0.5 AR process (Levinson): {a_lev.item()}")

if __name__ == "__main__":
    test_burg_gradient()
    test_levinson_gradient()
    test_burg_values()
