import torch
import numpy as np
from model import LMoments1d
import math

def scipy_l_moments(x):
    # Very simple implementation of first 4 L-moments for verification
    # L1 = E[X]
    # L2 = 1/2 E[X(2:2) - X(1:2)]
    # ... but it's easier to use PWMs
    # b_r = 1/n sum_{j=1}^n [comb(j-1, r) / comb(n-1, r)] x_(j)
    n = len(x)
    x_sorted = np.sort(x)

    def get_b(r):
        b = 0
        for j in range(1, n + 1):
            weight = math.comb(j-1, r) / math.comb(n-1, r)
            b += weight * x_sorted[j-1]
        return b / n

    b0 = get_b(0)
    b1 = get_b(1)
    b2 = get_b(2)
    b3 = get_b(3)

    L1 = b0
    L2 = 2*b1 - b0
    L3 = 6*b2 - 6*b1 + b0
    L4 = 20*b3 - 30*b2 + 12*b1 - b0

    return [L1, L2, L3, L4]

def test_l_moments_values():
    print("Testing L-moment values...")
    x_np = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    x_torch = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    layer = LMoments1d(max_order=4)
    l_moments = layer(x_torch).squeeze().detach().numpy()

    expected = scipy_l_moments(x_np)

    print(f"Computed: {l_moments}")
    print(f"Expected: {expected}")

    assert np.allclose(l_moments, expected, atol=1e-4)
    print("Value test passed!")

def test_gradients():
    print("Testing gradients...")
    x = torch.randn(2, 1, 10, requires_grad=True)
    layer = LMoments1d(max_order=4)
    l_moments = layer(x)

    loss = l_moments.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert torch.abs(x.grad).sum() > 0
    print("Gradient test passed!")

if __name__ == "__main__":
    test_l_moments_values()
    test_gradients()
