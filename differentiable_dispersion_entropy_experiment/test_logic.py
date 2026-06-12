import torch
import pytest
from model import DDELayer

def test_dde_shape():
    B, L = 4, 40
    c, m, tau = 3, 2, 1
    dde = DDELayer(c=c, m=m, tau=tau)
    x = torch.randn(B, L)
    out = dde(x)
    assert out.shape == (B, 1)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1.0001) # Allow some epsilon for numerical issues

def test_dde_differentiability():
    B, L = 2, 20
    c, m, tau = 3, 2, 1
    dde = DDELayer(c=c, m=m, tau=tau, learnable=True)
    x = torch.randn(B, L, requires_grad=True)
    out = dde(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert dde.centers.grad is not None
    assert dde.sigma.grad is not None

    print("Gradients with respect to input:", x.grad.abs().mean().item())
    print("Gradients with respect to centers:", dde.centers.grad.abs().mean().item())
    print("Gradients with respect to sigma:", dde.sigma.grad.abs().mean().item())

def test_dde_parameters():
    c, m, tau = 3, 2, 1
    dde = DDELayer(c=c, m=m, tau=tau, learnable=False)
    assert len(list(dde.parameters())) == 0

    dde_learnable = DDELayer(c=c, m=m, tau=tau, learnable=True)
    assert len(list(dde_learnable.parameters())) == 2 # centers and sigma

if __name__ == "__main__":
    test_dde_shape()
    test_dde_differentiability()
    test_dde_parameters()
    print("All logic tests passed!")
