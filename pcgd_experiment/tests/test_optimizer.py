import torch
import torch.nn as nn
from pcgd_experiment.optimizer import PCGDOptimizer, get_consensus_gradient

def test_get_consensus_gradient():
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Check if it runs without error
    grads = get_consensus_gradient(model, x, y)
    assert isinstance(grads, dict)
    for p in model.parameters():
        found = False
        for g in grads.values():
            if g.shape == p.shape:
                found = True
                break
        assert found

def test_optimizer_step():
    model = nn.Linear(10, 2)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt = PCGDOptimizer(model, base_opt)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    w_before = model.weight.clone()
    opt.zero_grad()
    opt.step(x, y)
    w_after = model.weight

    assert not torch.equal(w_before, w_after)

if __name__ == "__main__":
    test_get_consensus_gradient()
    test_optimizer_step()
    print("Tests passed!")
