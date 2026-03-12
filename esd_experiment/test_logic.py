import torch
import torch.nn as nn
from esd_lib import DecorrManager, compute_correlations, decorrelation_loss

def test_correlations():
    # Create features with known correlation
    # Feature 1 and 2 are highly correlated, Feature 3 is independent
    batch_size = 100
    f1 = torch.randn(batch_size, 1)
    f2 = f1 + 0.01 * torch.randn(batch_size, 1)
    f3 = torch.randn(batch_size, 1)

    features = torch.cat([f1, f2, f3], dim=1)
    corr = compute_correlations(features)

    print("Correlation matrix:\n", corr)
    assert corr.shape == (3, 3)
    assert torch.allclose(torch.diag(corr), torch.ones(3))
    assert corr[0, 1] > 0.9
    assert abs(corr[0, 2]) < 0.3

def test_decorr_manager_activation():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    manager = DecorrManager(model, mode='Decorr')

    x = torch.randn(8, 10)
    output = model(x)

    loss = manager.compute_loss()
    print("Activation Decorr Loss:", loss.item())
    assert loss > 0

    manager.clear()
    assert len(manager.captured_tensors) == 0

def test_decorr_manager_esd():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    manager = DecorrManager(model, mode='ESD')

    x = torch.randn(8, 10)
    output = model(x)
    target = torch.randn(8, 5)
    ce_loss = nn.MSELoss()(output, target)

    esd_loss = manager.compute_loss(ce_loss)
    print("ESD Loss:", esd_loss.item())
    assert esd_loss > 0

    # Check if we can backprop through ESD loss
    esd_loss.backward()
    for p in model.parameters():
        assert p.grad is not None

if __name__ == "__main__":
    print("Running test_correlations...")
    test_correlations()
    print("Running test_decorr_manager_activation...")
    test_decorr_manager_activation()
    print("Running test_decorr_manager_esd...")
    test_decorr_manager_esd()
    print("All tests passed!")
