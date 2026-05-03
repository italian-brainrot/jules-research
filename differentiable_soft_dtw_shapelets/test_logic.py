import torch
import torch.nn as nn
import time
from model import SoftDTWShapeletLayer, EuclideanShapeletLayer

def test_layer_shapes():
    B, C, L = 8, 1, 40
    K = 10
    NS = 16
    x = torch.randn(B, C, L)

    # Test Euclidean
    layer_e = EuclideanShapeletLayer(C, NS, K)
    out_e = layer_e(x)
    assert out_e.shape == (B, NS)

    # Test Soft-DTW
    layer_s = SoftDTWShapeletLayer(C, NS, K)
    start = time.time()
    out_s = layer_s(x)
    end = time.time()
    assert out_s.shape == (B, NS)
    print(f"Soft-DTW forward took {end - start:.4f}s for batch of {B}")

    loss = out_s.sum()
    start = time.time()
    loss.backward()
    end = time.time()
    print(f"Soft-DTW backward took {end - start:.4f}s for batch of {B}")

    assert layer_s.shapelets.grad is not None
    assert layer_s.gamma.grad is not None

def test_differentiability_wrt_gamma():
    B, C, L = 2, 1, 10
    K = 5
    NS = 2
    x = torch.randn(B, C, L)
    gamma = torch.tensor(1.0, requires_grad=True)

    layer = SoftDTWShapeletLayer(C, NS, K, gamma=1.0)
    # Re-assign gamma to the one with requires_grad
    layer.gamma = nn.Parameter(gamma)

    out = layer(x)
    loss = out.mean()
    loss.backward()

    assert layer.gamma.grad is not None
    print("Gamma grad:", layer.gamma.grad.item())

def test_shift_invariance():
    # DTW should be more robust to shifts than Euclidean distance
    B, C, L = 1, 1, 20
    K = 5
    NS = 1

    x = torch.zeros(B, C, L)
    x[0, 0, 5:10] = 1.0 # a bump

    layer_e = EuclideanShapeletLayer(C, NS, K)
    layer_s = SoftDTWShapeletLayer(C, NS, K)

    # Set shapelet to the same bump
    with torch.no_grad():
        bump = torch.ones(1, 1, K)
        layer_e.shapelets.copy_(bump)
        layer_s.shapelets.copy_(bump)

    # Distances for original
    d_e_orig = layer_e(x)
    d_s_orig = layer_s(x)

    # Shifted signal
    x_shifted = torch.zeros(B, C, L)
    x_shifted[0, 0, 6:11] = 1.0

    d_e_shift = layer_e(x_shifted)
    d_s_shift = layer_s(x_shifted)

    print(f"Euclidean: orig={d_e_orig.item():.4f}, shift={d_e_shift.item():.4f}, diff={abs(d_e_orig - d_e_shift).item():.4f}")
    print(f"Soft-DTW: orig={d_s_orig.item():.4f}, shift={d_s_shift.item():.4f}, diff={abs(d_s_orig - d_s_shift).item():.4f}")
    # Note: Soft-DTW with sliding window and soft-min pooling already has some shift invariance from the windowing.
    # But DTW within the window adds more.

if __name__ == "__main__":
    test_layer_shapes()
    test_differentiability_wrt_gamma()
    test_shift_invariance()
