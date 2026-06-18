import torch
import pytest
from nca_loss import nca_loss

def test_nca_loss_basic():
    embeddings = torch.randn(10, 5, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    loss = nca_loss(embeddings, targets)
    assert loss.item() > 0
    loss.backward()
    assert embeddings.grad is not None
    assert not torch.isnan(embeddings.grad).any()

def test_nca_loss_single_class():
    embeddings = torch.randn(5, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4])
    loss = nca_loss(embeddings, targets)
    # Should be 0 since no pairs of same class
    assert loss.item() == 0
    # If loss is a constant 0, it might not have grad_fn if we created it as a new tensor
    if loss.grad_fn is not None:
        loss.backward()
        assert embeddings.grad is not None

def test_nca_loss_large_temp():
    embeddings = torch.randn(10, 5, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    loss = nca_loss(embeddings, targets, temperature=1e6)
    assert loss.item() > 0
    loss.backward()
    assert not torch.isnan(embeddings.grad).any()

def test_nca_loss_small_temp():
    embeddings = torch.randn(10, 5, requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    loss = nca_loss(embeddings, targets, temperature=1e-6)
    assert loss.item() > 0
    loss.backward()
    assert not torch.isnan(embeddings.grad).any()

if __name__ == "__main__":
    pytest.main([__file__])
