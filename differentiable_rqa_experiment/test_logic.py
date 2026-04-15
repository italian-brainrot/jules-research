import torch
from model import DRQALayer

def test_drqa_differentiability():
    layer = DRQALayer()
    x = torch.randn(4, 40, requires_grad=True)
    out = layer(x)

    assert out.shape == (4, 3)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.log_eps.grad is not None
    assert layer.log_gamma.grad is not None
    print("Differentiability test passed!")

def test_drqa_logic():
    # Constant signal should have high recurrence
    layer = DRQALayer(eps=0.1, gamma=100.0)
    x = torch.zeros(2, 10)
    out = layer(x)
    # RR should be high (close to 1.0)
    assert torch.all(out[:, 0] > 0.9)
    # DET should be high for constant signal
    assert torch.all(out[:, 1] > 0.9)
    # LAM should be high for constant signal
    assert torch.all(out[:, 2] > 0.9)

    # Random signal should have lower recurrence than constant
    x_rand = torch.randn(2, 10) * 10.0
    out_rand = layer(x_rand)
    assert torch.all(out_rand[:, 0] < out[:, 0])

    print("Logic test passed!")

if __name__ == "__main__":
    test_drqa_differentiability()
    test_drqa_logic()
