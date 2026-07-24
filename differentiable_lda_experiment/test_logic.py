import torch
from differentiable_lda_experiment.lda import DLDALoss

def test_lda_loss_gradients():
    B, D = 32, 8
    features = torch.randn(B, D, requires_grad=True)
    targets = torch.randint(0, 3, (B,))

    lda_loss_fn = DLDALoss()
    loss = lda_loss_fn(features, targets)

    assert torch.isfinite(loss)
    loss.backward()

    assert features.grad is not None
    assert torch.all(torch.isfinite(features.grad))
    print("Gradient test passed!")

def test_lda_loss_edge_cases():
    B, D = 32, 8
    features = torch.randn(B, D, requires_grad=True)

    # Single class case
    targets_single = torch.zeros(B, dtype=torch.long)
    lda_loss_fn = DLDALoss()
    loss = lda_loss_fn(features, targets_single)
    assert loss == 0.0
    assert loss.requires_grad
    loss.backward()
    print("Single class test passed!")

if __name__ == "__main__":
    test_lda_loss_gradients()
    test_lda_loss_edge_cases()
