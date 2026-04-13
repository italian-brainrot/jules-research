import torch
from model import HjorthLayer

def test_hjorth_forward():
    batch_size = 4
    length = 100
    x = torch.randn(batch_size, length)
    layer = HjorthLayer()
    features = layer(x)

    assert features.shape == (batch_size, 3), f"Expected shape (4, 3), got {features.shape}"
    assert torch.all(features[:, 0] >= 0), "Activity should be non-negative"
    assert torch.all(features[:, 1] >= 0), "Mobility should be non-negative"
    assert torch.all(features[:, 2] >= 0), "Complexity should be non-negative"
    print("Forward pass test passed.")

def test_hjorth_gradients():
    batch_size = 4
    length = 100
    x = torch.randn(batch_size, length, requires_grad=True)
    layer = HjorthLayer()
    features = layer(x)
    loss = features.sum()
    loss.backward()

    assert x.grad is not None, "Gradients should not be None"
    assert not torch.isnan(x.grad).any(), "Gradients should not contain NaNs"
    print("Gradient test passed.")

def test_hjorth_invariance():
    # Hjorth parameters (except Activity) should be scale invariant if we ignore eps
    # Actually, Mobility = sqrt(var(dx)/var(x)), so it is scale invariant.
    # Complexity is also scale invariant.
    batch_size = 1
    length = 100
    x = torch.randn(batch_size, length)
    layer = HjorthLayer(eps=0)

    feat1 = layer(x)
    feat2 = layer(x * 2.0)

    # Activity is variance, so it should scale by 4.0
    assert torch.allclose(feat1[:, 0] * 4.0, feat2[:, 0]), "Activity should scale with variance"
    assert torch.allclose(feat1[:, 1], feat2[:, 1]), "Mobility should be scale invariant"
    assert torch.allclose(feat1[:, 2], feat2[:, 2]), "Complexity should be scale invariant"
    print("Scale invariance test passed.")

if __name__ == "__main__":
    test_hjorth_forward()
    test_hjorth_gradients()
    test_hjorth_invariance()
