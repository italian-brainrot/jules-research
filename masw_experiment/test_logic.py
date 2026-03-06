import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MLP
from masw_utils import compute_masw_gradients

def test_masw_logic():
    device = 'cpu'
    model = MLP(40, [32], 10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Create fake data
    inputs = torch.randn(8, 40).to(device)
    targets = torch.randint(0, 10, (8,)).to(device)

    # 1. Test without momentum (first step)
    print("Testing MASW without momentum...")
    grads = compute_masw_gradients(model, optimizer, inputs, targets, gamma=1.0)
    assert len(grads) == len(list(model.parameters()))
    for g, p in zip(grads, model.parameters()):
        assert g.shape == p.shape
        assert not torch.isnan(g).any()
    print("Passed!")

    # 2. Populate momentum
    print("Populating momentum...")
    out = model(inputs)
    loss = torch.nn.functional.cross_entropy(out, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 3. Test with momentum
    print("Testing MASW with momentum...")
    grads_with_m = compute_masw_gradients(model, optimizer, inputs, targets, gamma=2.0)
    assert len(grads_with_m) == len(list(model.parameters()))
    for g, p in zip(grads_with_m, model.parameters()):
        assert g.shape == p.shape
        assert not torch.isnan(g).any()
    print("Passed!")

    # 4. Verify weighting effect (qualitative)
    # If we set gamma to 0, it should be equivalent to standard mean gradient
    print("Testing gamma=0 (should be equivalent to mean gradient)...")
    optimizer.zero_grad()
    out = model(inputs)
    loss = torch.nn.functional.cross_entropy(out, targets)
    loss.backward()
    true_mean_grads = [p.grad.clone() for p in model.parameters()]
    optimizer.zero_grad()

    masw_grads_gamma0 = compute_masw_gradients(model, optimizer, inputs, targets, gamma=0.0)
    for g, mg in zip(masw_grads_gamma0, true_mean_grads):
        assert torch.allclose(g, mg, atol=1e-5)
    print("Passed!")

if __name__ == "__main__":
    test_masw_logic()
