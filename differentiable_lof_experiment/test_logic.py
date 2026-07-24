import torch
from differentiable_lof_experiment.model import DLOFLoss, LOFRegularizedMLP

def test_dlof_loss_gradient():
    batch_size = 10
    dim = 8
    k = 3
    x = torch.randn(batch_size, dim, requires_grad=True)
    loss_fn = DLOFLoss(k=k)
    loss = loss_fn(x)

    assert loss.item() >= 0
    loss.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

def test_lof_regularized_mlp_forward():
    input_dim = 40
    batch_size = 16
    model = LOFRegularizedMLP(input_dim=input_dim, k=5)
    x = torch.randn(batch_size, input_dim)

    # Training mode
    model.train()
    out, lof_loss = model(x, return_lof=True)
    assert out.shape == (batch_size, 10)
    assert lof_loss.item() >= 0

    # Eval mode
    model.eval()
    out = model(x)
    assert out.shape == (batch_size, 10)

def test_dlof_extreme_outlier():
    # Test if an outlier actually increases the loss
    k = 3
    x = torch.randn(10, 2) * 0.1 # Cluster around origin
    outlier = torch.tensor([[10.0, 10.0]]) # Far away outlier

    loss_fn = DLOFLoss(k=k)

    loss_no_outlier = loss_fn(x)

    x_with_outlier = torch.cat([x, outlier], dim=0)
    loss_with_outlier = loss_fn(x_with_outlier)

    # LOF of an outlier should be >> 1, so (LOF-1)^2 should be large
    assert loss_with_outlier > loss_no_outlier
