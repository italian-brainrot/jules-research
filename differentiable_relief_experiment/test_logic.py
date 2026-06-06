import torch
import pytest
from differentiable_relief_experiment.layer import DReliefLoss
from differentiable_relief_experiment.model import get_model

def test_drelief_loss_gradient():
    B, D = 16, 32
    features = torch.randn(B, D, requires_grad=True)
    targets = torch.randint(0, 5, (B,))

    # Ensure there is at least one pair of same class and different class
    targets[0] = 0
    targets[1] = 0
    targets[2] = 1

    loss_fn = DReliefLoss(temperature=0.1)
    loss = loss_fn(features, targets)

    assert not torch.isnan(loss)
    loss.backward()
    assert features.grad is not None
    assert not torch.all(features.grad == 0)
    print(f"Loss: {loss.item()}, Gradient norm: {features.grad.norm().item()}")

def test_drelief_loss_behavior():
    # If all same class, dist_hit should be minimized?
    # Actually if all same class, num_classes-1 = 0.
    # Let's test with 2 classes.
    features = torch.tensor([[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]], requires_grad=True)
    targets = torch.tensor([0, 0, 1, 1])

    loss_fn = DReliefLoss(temperature=0.01) # Low temperature makes it almost like hard Relief
    loss = loss_fn(features, targets)

    # Hits: (0,1), (2,3) - very close
    # Misses: (0,2), (0,3), (1,2), (1,3) - far
    # Hit distances should be small, Miss distances should be large.
    # Loss = Avg(HitDist - MissDist) should be negative and large.
    print(f"Loss with good clustering: {loss.item()}")

    features_bad = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.1, 0.0], [0.1, 1.1]], requires_grad=True)
    targets_bad = torch.tensor([0, 0, 1, 1])
    # Hits: (0,1) - far, (2,3) - far
    # Misses: (0,2) - close, (1,3) - close
    loss_bad = loss_fn(features_bad, targets_bad)
    print(f"Loss with bad clustering: {loss_bad.item()}")

    assert loss.item() < loss_bad.item()

def test_model_forward():
    model = get_model(input_dim=40, hidden_dim=128, output_dim=10)
    x = torch.randn(8, 40)
    logits, feat = model(x, return_features=True)
    assert logits.shape == (8, 10)
    assert feat.shape == (8, 128)

if __name__ == "__main__":
    test_drelief_loss_gradient()
    test_drelief_loss_behavior()
    test_model_forward()
    print("All tests passed!")
